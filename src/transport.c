/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <self_amdgpu_runtime/runtime.h>
#include <self_amdgpu_runtime/code_object.h>

#include "transport_internal.h"
#include "sha256_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define SAGR_INSTANCE_MAGIC UINT64_C(0x53414752494e5354)
#define SAGR_QUEUE_MAGIC UINT64_C(0x5341475251554555)
#define SAGR_MEMORY_MAGIC UINT64_C(0x534147524d454d59)
#define SAGR_SIGNAL_MAGIC UINT64_C(0x534147525349474e)
#define SAGR_MEMORY_SIMULATED_VA_BASE UINT64_C(0x0000100000000000)
#define SAGR_MEMORY_SIMULATED_VA_STRIDE UINT64_C(2147483648)

struct sagr_queue;
struct sagr_memory;
struct sagr_signal;
struct sagr_pending_dispatch;

struct sagr_instance {
  uint64_t magic;
  int socket_fd;
  uint64_t next_request_id;
  sagr_instance_info_t info;
  struct sagr_queue *queues;
  uint32_t queue_count;
  struct sagr_memory *memories;
  uint32_t memory_count;
  struct sagr_signal *signals;
  uint32_t signal_count;
  uint32_t pending_signal_wait_count;
  uint64_t last_signal_generation;
  struct sagr_pending_dispatch *pending_dispatch;
  int operation_active;
  int transport_poisoned;
};

struct sagr_signal {
  uint64_t magic;
  struct sagr_instance *instance;
  uint64_t signal_id;
  uint64_t generation;
  int64_t value;
  uint64_t next_sequence;
  int wait_pending;
  uint64_t wait_sequence;
  uint64_t wait_condition;
  uint64_t wait_compare_bits;
  uint64_t wait_request_id;
  uint64_t wait_ack_value_bits;
  uint64_t wait_ack_tick;
  uint64_t wait_ready;
  int wait_trigger_known;
  uint64_t wait_trigger_value_bits;
  uint64_t wait_trigger_tick;
  int completion_buffered;
  sagr_wire_signal_response_t buffered_completion;
  struct sagr_signal *next;
};

struct sagr_memory {
  uint64_t magic;
  struct sagr_instance *instance;
  uint64_t allocation_id;
  uint64_t generation;
  uint64_t simulated_va;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  struct sagr_memory *next;
};

struct sagr_queue {
  uint64_t magic;
  struct sagr_instance *instance;
  uint64_t queue_id;
  uint64_t generation;
  uint64_t next_sequence;
  uint32_t depth;
  uint32_t pending_count;
  uint64_t pending_sequences[SAGR_QUEUE_MAX_INFLIGHT];
  uint64_t pending_kinds[SAGR_QUEUE_MAX_INFLIGHT];
  uint64_t pending_request_ids[SAGR_QUEUE_MAX_INFLIGHT];
  uint64_t pending_ack_ticks[SAGR_QUEUE_MAX_INFLIGHT];
  uint32_t buffered_count;
  sagr_wire_queue_response_t buffered[SAGR_QUEUE_MAX_INFLIGHT];
  struct sagr_queue *next;
};

struct sagr_pending_dispatch {
  struct sagr_queue *queue;
  struct sagr_memory *input;
  struct sagr_memory *output;
  struct sagr_signal *signal;
  uint64_t request_id;
  uint64_t expected_signal_value_bits;
  uint64_t signal_completion_value_bits;
  uint64_t signal_completion_tick;
  sagr_wire_dispatch_response_t ack;
  int signal_completion_seen;
  int completion_buffered;
  int dispatch_completion_consumed;
  int signal_completion_consumed;
  sagr_wire_dispatch_response_t buffered_completion;
};

typedef struct monotonic_deadline {
  int infinite;
  struct timespec time;
} monotonic_deadline_t;

static int bytes_are_zero(const uint8_t *bytes, size_t size) {
  uint8_t combined = 0;
  size_t index;
  for (index = 0; index < size; ++index) {
    combined = (uint8_t)(combined | bytes[index]);
  }
  return combined == 0;
}

static int reserved_is_zero(const uint8_t *bytes, size_t size) {
  return bytes_are_zero(bytes, size);
}

static int capabilities_subset(const uint64_t *subset,
                               const uint64_t *superset) {
  uint32_t index;
  for (index = 0; index < SAGR_CAPABILITY_WORD_COUNT; ++index) {
    if ((subset[index] & ~superset[index]) != 0) {
      return 0;
    }
  }
  return 1;
}

static void initialize_error(sagr_error_info_t *error, uint32_t error_size) {
  size_t clear_size;
  if (error == NULL) {
    return;
  }
  clear_size = error_size < sizeof(*error) ? error_size : sizeof(*error);
  memset(error, 0, clear_size);
  if (error_size >= sizeof(error->struct_size)) {
    error->struct_size = (uint32_t)sizeof(*error);
  }
  if (error_size < sizeof(*error)) {
    return;
  }
  error->status = SAGR_STATUS_SUCCESS;
  error->wire_status = -1;
}

static sagr_status_t fail_open(sagr_error_info_t *error, uint32_t error_size,
                               sagr_status_t status, int32_t wire_status,
                               int native_errno, const char *message) {
  if (error != NULL && error_size >= sizeof(*error)) {
    error->status = status;
    error->wire_status = wire_status;
    error->native_errno = native_errno;
    if (message != NULL) {
      (void)snprintf(error->message, sizeof(error->message), "%s", message);
    }
  }
  return status;
}

static sagr_status_t validate_options(
    const sagr_instance_open_options_t *options) {
  uint32_t minimum;
  uint32_t maximum;
  const int queue_offered =
      (options->offered_capabilities[SAGR_CAPABILITY_QUEUE_WORD] &
       SAGR_CAPABILITY_QUEUE_MASK) != 0;
  const int queue_required =
      (options->required_capabilities[SAGR_CAPABILITY_QUEUE_WORD] &
       SAGR_CAPABILITY_QUEUE_MASK) != 0;
  const int memory_offered =
      (options->offered_capabilities[SAGR_CAPABILITY_MEMORY_WORD] &
       SAGR_CAPABILITY_MEMORY_MASK) != 0;
  const int memory_required =
      (options->required_capabilities[SAGR_CAPABILITY_MEMORY_WORD] &
       SAGR_CAPABILITY_MEMORY_MASK) != 0;
  const int signal_offered =
      (options->offered_capabilities[SAGR_CAPABILITY_SIGNAL_WORD] &
       SAGR_CAPABILITY_SIGNAL_MASK) != 0;
  const int signal_required =
      (options->required_capabilities[SAGR_CAPABILITY_SIGNAL_WORD] &
       SAGR_CAPABILITY_SIGNAL_MASK) != 0;
  const int dispatch_offered =
      (options->offered_capabilities[SAGR_CAPABILITY_DISPATCH_WORD] &
       SAGR_CAPABILITY_DISPATCH_MASK) != 0;
  const int dispatch_required =
      (options->required_capabilities[SAGR_CAPABILITY_DISPATCH_WORD] &
       SAGR_CAPABILITY_DISPATCH_MASK) != 0;
  const int kmt_offered =
      (options->offered_capabilities[SAGR_CAPABILITY_KMT_WORD] &
       SAGR_CAPABILITY_KMT_MASK) != 0;
  const int kmt_required =
      (options->required_capabilities[SAGR_CAPABILITY_KMT_WORD] &
       SAGR_CAPABILITY_KMT_MASK) != 0;
  const int code_object_offered =
      (options->offered_capabilities[SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_WORD] &
       SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK) != 0;
  const int code_object_required =
      (options->required_capabilities[SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_WORD] &
       SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK) != 0;
  if (options->struct_size < sizeof(*options) || options->flags != 0 ||
      options->cancel_fd < -1 || options->reserved0 != 0 ||
      !reserved_is_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  minimum = ((uint32_t)options->minimum_version_major << 16) |
            options->minimum_version_minor;
  maximum = ((uint32_t)options->maximum_version_major << 16) |
            options->maximum_version_minor;
  if (minimum > maximum ||
      !capabilities_subset(options->required_capabilities,
                           options->offered_capabilities) ||
      (options->offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] &
       SAGR_CAPABILITY_TOPOLOGY_MASK) == 0 ||
      (options->required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] &
      SAGR_CAPABILITY_TOPOLOGY_MASK) == 0 ||
      queue_offered != queue_required || memory_offered != memory_required ||
      signal_offered != signal_required ||
      dispatch_offered != dispatch_required ||
      kmt_offered != kmt_required ||
      code_object_offered != code_object_required ||
      (dispatch_required != 0 &&
       (queue_required == 0 || memory_required == 0 ||
        signal_required == 0))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_memory_allocate_options(
    const sagr_memory_allocate_options_t *options) {
  if (options->struct_size < sizeof(*options) || options->flags != 0 ||
      options->size_bytes == 0 ||
      (options->alignment_bytes != SAGR_MEMORY_ALIGNMENT_4K &&
       options->alignment_bytes != SAGR_MEMORY_ALIGNMENT_64K) ||
      !reserved_is_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_memory_operation_options(
    const sagr_memory_operation_options_t *options) {
  if (options->struct_size < sizeof(*options) || options->flags != 0 ||
      options->cancel_fd < -1 || options->reserved0 != 0 ||
      !reserved_is_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_queue_create_options(
    const sagr_queue_create_options_t *options) {
  if (options->struct_size < sizeof(*options) || options->flags != 0 ||
      options->depth == 0 || options->depth > SAGR_QUEUE_MAX_DEPTH ||
      options->reserved0 != 0 ||
      !reserved_is_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_queue_operation_options(
    const sagr_queue_operation_options_t *options) {
  if (options->struct_size < sizeof(*options) || options->flags != 0 ||
      options->cancel_fd < -1 || options->reserved0 != 0 ||
      !reserved_is_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_signal_create_options(
    const sagr_signal_create_options_t *options) {
  if (options->struct_size < sizeof(*options) || options->flags != 0 ||
      !reserved_is_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_signal_operation_options(
    const sagr_signal_operation_options_t *options) {
  if (options->struct_size < sizeof(*options) || options->flags != 0 ||
      options->cancel_fd < -1 || options->reserved0 != 0 ||
      !reserved_is_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_pinned_dispatch_options(
    const sagr_pinned_dispatch_options_t *options) {
  if (options->struct_size < sizeof(*options) || options->flags != 0 ||
      options->fixture_id != SAGR_DISPATCH_FIXTURE_GFX950_XOR_U8_V1 ||
      !reserved_is_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

static int assign_time_t(uint64_t seconds, time_t *destination) {
  const time_t converted = (time_t)seconds;
  if ((uint64_t)converted != seconds) {
    errno = EOVERFLOW;
    return -1;
  }
  *destination = converted;
  return 0;
}

static int make_deadline(uint64_t timeout_ns, uint64_t absolute_deadline_ns,
                         monotonic_deadline_t *deadline) {
  uint64_t seconds;
  uint64_t nanoseconds;
  uint64_t target_seconds;
  if (absolute_deadline_ns != 0) {
    deadline->infinite = 0;
    seconds = absolute_deadline_ns / UINT64_C(1000000000);
    nanoseconds = absolute_deadline_ns % UINT64_C(1000000000);
    if (assign_time_t(seconds, &deadline->time.tv_sec) != 0) {
      return -1;
    }
    deadline->time.tv_nsec = (long)nanoseconds;
    return 0;
  }
  if (timeout_ns == UINT64_MAX) {
    deadline->infinite = 1;
    memset(&deadline->time, 0, sizeof(deadline->time));
    return 0;
  }
  if (timeout_ns == 0) {
    timeout_ns = SAGR_DEFAULT_OPEN_TIMEOUT_NS;
  }
  deadline->infinite = 0;
  if (clock_gettime(CLOCK_MONOTONIC, &deadline->time) != 0) {
    return -1;
  }
  if (deadline->time.tv_sec < 0) {
    errno = EOVERFLOW;
    return -1;
  }
  seconds = timeout_ns / UINT64_C(1000000000);
  nanoseconds = timeout_ns % UINT64_C(1000000000);
  if (seconds > UINT64_MAX - (uint64_t)deadline->time.tv_sec) {
    errno = EOVERFLOW;
    return -1;
  }
  target_seconds = (uint64_t)deadline->time.tv_sec + seconds;
  nanoseconds += (uint64_t)deadline->time.tv_nsec;
  if (nanoseconds >= UINT64_C(1000000000)) {
    if (target_seconds == UINT64_MAX) {
      errno = EOVERFLOW;
      return -1;
    }
    ++target_seconds;
    nanoseconds -= UINT64_C(1000000000);
  }
  if (assign_time_t(target_seconds, &deadline->time.tv_sec) != 0) {
    return -1;
  }
  deadline->time.tv_nsec = (long)nanoseconds;
  return 0;
}

static sagr_status_t prepare_queue_operation(
    const sagr_queue_operation_options_t *options,
    sagr_queue_operation_options_t *local_options,
    monotonic_deadline_t *deadline, int *native_errno) {
  sagr_status_t status;
  if (options == NULL) {
    status = sagr_queue_operation_options_init(
        local_options, (uint32_t)sizeof(*local_options));
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
  } else {
    if (options->struct_size < sizeof(*options)) {
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
    memcpy(local_options, options, sizeof(*local_options));
  }
  status = validate_queue_operation_options(local_options);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (local_options->cancel_fd >= 0) {
    const int flags = fcntl(local_options->cancel_fd, F_GETFD);
    if (flags < 0) {
      *native_errno = errno;
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
    if ((flags & FD_CLOEXEC) == 0) {
      *native_errno = EINVAL;
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
  }
  if (make_deadline(local_options->timeout_ns,
                    local_options->absolute_deadline_ns, deadline) != 0) {
    *native_errno = errno;
    return errno == EOVERFLOW ? SAGR_STATUS_INVALID_ARGUMENT
                              : SAGR_STATUS_INTERNAL_ERROR;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t prepare_memory_operation(
    const sagr_memory_operation_options_t *options,
    sagr_memory_operation_options_t *local_options,
    monotonic_deadline_t *deadline, int *native_errno) {
  sagr_status_t status;
  if (options == NULL) {
    status = sagr_memory_operation_options_init(
        local_options, (uint32_t)sizeof(*local_options));
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
  } else {
    if (options->struct_size < sizeof(*options)) {
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
    memcpy(local_options, options, sizeof(*local_options));
  }
  status = validate_memory_operation_options(local_options);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (local_options->cancel_fd >= 0) {
    const int flags = fcntl(local_options->cancel_fd, F_GETFD);
    if (flags < 0) {
      *native_errno = errno;
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
    if ((flags & FD_CLOEXEC) == 0) {
      *native_errno = EINVAL;
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
  }
  if (make_deadline(local_options->timeout_ns,
                    local_options->absolute_deadline_ns, deadline) != 0) {
    *native_errno = errno;
    return errno == EOVERFLOW ? SAGR_STATUS_INVALID_ARGUMENT
                              : SAGR_STATUS_INTERNAL_ERROR;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t prepare_signal_operation(
    const sagr_signal_operation_options_t *options,
    sagr_signal_operation_options_t *local_options,
    monotonic_deadline_t *deadline, int *native_errno) {
  sagr_status_t status;
  if (options == NULL) {
    status = sagr_signal_operation_options_init(
        local_options, (uint32_t)sizeof(*local_options));
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
  } else {
    if (options->struct_size < sizeof(*options)) {
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
    memcpy(local_options, options, sizeof(*local_options));
  }
  status = validate_signal_operation_options(local_options);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (local_options->cancel_fd >= 0) {
    const int flags = fcntl(local_options->cancel_fd, F_GETFD);
    if (flags < 0) {
      *native_errno = errno;
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
    if ((flags & FD_CLOEXEC) == 0) {
      *native_errno = EINVAL;
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
  }
  if (make_deadline(local_options->timeout_ns,
                    local_options->absolute_deadline_ns, deadline) != 0) {
    *native_errno = errno;
    return errno == EOVERFLOW ? SAGR_STATUS_INVALID_ARGUMENT
                              : SAGR_STATUS_INTERNAL_ERROR;
  }
  return SAGR_STATUS_SUCCESS;
}

static int deadline_remaining_milliseconds(const monotonic_deadline_t *deadline,
                                           int *milliseconds) {
  struct timespec now;
  int64_t seconds;
  int64_t nanoseconds;
  uint64_t total_ns;
  uint64_t rounded_ms;
  if (deadline->infinite != 0) {
    *milliseconds = -1;
    return 1;
  }
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    return -1;
  }
  seconds = (int64_t)deadline->time.tv_sec - (int64_t)now.tv_sec;
  nanoseconds = (int64_t)deadline->time.tv_nsec - (int64_t)now.tv_nsec;
  if (nanoseconds < 0) {
    --seconds;
    nanoseconds += INT64_C(1000000000);
  }
  if (seconds < 0 || (seconds == 0 && nanoseconds <= 0)) {
    *milliseconds = 0;
    return 0;
  }
  if ((uint64_t)seconds > UINT64_MAX / UINT64_C(1000000000)) {
    *milliseconds = INT_MAX;
    return 1;
  }
  total_ns = (uint64_t)seconds * UINT64_C(1000000000) +
             (uint64_t)nanoseconds;
  rounded_ms = total_ns / UINT64_C(1000000);
  if (total_ns % UINT64_C(1000000) != 0) {
    ++rounded_ms;
  }
  *milliseconds = rounded_ms > (uint64_t)INT_MAX ? INT_MAX : (int)rounded_ms;
  return 1;
}

static sagr_status_t check_deadline(const monotonic_deadline_t *deadline) {
  int milliseconds;
  const int remaining =
      deadline_remaining_milliseconds(deadline, &milliseconds);
  if (remaining < 0) {
    return SAGR_STATUS_INTERNAL_ERROR;
  }
  return remaining == 0 ? SAGR_STATUS_TIMED_OUT : SAGR_STATUS_SUCCESS;
}

static sagr_status_t check_operation_state(
    const monotonic_deadline_t *deadline, int cancel_fd, int *native_errno) {
  sagr_status_t status = check_deadline(deadline);
  if (status != SAGR_STATUS_SUCCESS || cancel_fd < 0) {
    return status;
  }
  for (;;) {
    struct pollfd descriptor;
    int result;
    descriptor.fd = cancel_fd;
    descriptor.events = POLLIN;
    descriptor.revents = 0;
    result = poll(&descriptor, 1, 0);
    if (result > 0) {
      if ((descriptor.revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0) {
        *native_errno = (descriptor.revents & POLLNVAL) != 0 ? EBADF : 0;
        return SAGR_STATUS_CANCELLED;
      }
      return SAGR_STATUS_SUCCESS;
    }
    if (result == 0) {
      return SAGR_STATUS_SUCCESS;
    }
    if (errno != EINTR) {
      *native_errno = errno;
      return SAGR_STATUS_INTERNAL_ERROR;
    }
    status = check_deadline(deadline);
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
  }
}

static sagr_status_t wait_for_socket(int socket_fd, short events,
                                     const monotonic_deadline_t *deadline,
                                     int cancel_fd,
                                     int *native_errno) {
  for (;;) {
    struct pollfd descriptors[2];
    nfds_t descriptor_count = 1;
    int timeout_ms;
    int result;
    const int remaining =
        deadline_remaining_milliseconds(deadline, &timeout_ms);
    if (remaining < 0) {
      *native_errno = errno;
      return SAGR_STATUS_INTERNAL_ERROR;
    }
    if (remaining == 0) {
      return SAGR_STATUS_TIMED_OUT;
    }
    descriptors[0].fd = socket_fd;
    descriptors[0].events = events;
    descriptors[0].revents = 0;
    if (cancel_fd >= 0) {
      descriptors[1].fd = cancel_fd;
      descriptors[1].events = POLLIN;
      descriptors[1].revents = 0;
      descriptor_count = 2;
    }
    result = poll(descriptors, descriptor_count, timeout_ms);
    if (result > 0) {
      sagr_status_t state = check_deadline(deadline);
      if (state != SAGR_STATUS_SUCCESS) {
        return state;
      }
      if (descriptor_count == 2 &&
          (descriptors[1].revents &
           (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0) {
        *native_errno =
            (descriptors[1].revents & POLLNVAL) != 0 ? EBADF : 0;
        return SAGR_STATUS_CANCELLED;
      }
      if ((descriptors[0].revents & POLLNVAL) != 0) {
        *native_errno = EBADF;
        return SAGR_STATUS_CONNECTION_LOST;
      }
      if ((descriptors[0].revents &
           (short)(events | POLLERR | POLLHUP)) != 0) {
        return SAGR_STATUS_SUCCESS;
      }
      continue;
    }
    if (result == 0) {
      return SAGR_STATUS_TIMED_OUT;
    }
    if (errno != EINTR) {
      *native_errno = errno;
      return SAGR_STATUS_INTERNAL_ERROR;
    }
  }
}

static sagr_status_t map_connect_errno(int error_number) {
  switch (error_number) {
    case ENOENT:
    case ENOTDIR:
      return SAGR_STATUS_ENDPOINT_NOT_FOUND;
    case EACCES:
    case EPERM:
      return SAGR_STATUS_UNAUTHORIZED;
    case ETIMEDOUT:
      return SAGR_STATUS_TIMED_OUT;
    case ENOMEM:
    case ENOBUFS:
    case EMFILE:
    case ENFILE:
      return SAGR_STATUS_OUT_OF_RESOURCES;
    case ECANCELED:
      return SAGR_STATUS_CANCELLED;
    default:
      return SAGR_STATUS_UNAVAILABLE;
  }
}

static sagr_status_t connect_once(int socket_fd,
                                  const struct sockaddr_un *address,
                                  socklen_t address_size,
                                  const monotonic_deadline_t *deadline,
                                  int cancel_fd,
                                  int *native_errno) {
  for (;;) {
    int socket_error = 0;
    socklen_t socket_error_size = (socklen_t)sizeof(socket_error);
    sagr_status_t wait_status;
    wait_status = check_operation_state(deadline, cancel_fd, native_errno);
    if (wait_status != SAGR_STATUS_SUCCESS) {
      return wait_status;
    }
    if (connect(socket_fd, (const struct sockaddr *)address, address_size) ==
            0 ||
        errno == EISCONN) {
      return SAGR_STATUS_SUCCESS;
    }
    if (errno != EINPROGRESS && errno != EALREADY && errno != EINTR) {
      *native_errno = errno;
      return map_connect_errno(errno);
    }
    wait_status = wait_for_socket(socket_fd, POLLOUT, deadline, cancel_fd,
                                  native_errno);
    if (wait_status != SAGR_STATUS_SUCCESS) {
      return wait_status;
    }
    if (getsockopt(socket_fd, SOL_SOCKET, SO_ERROR, &socket_error,
                   &socket_error_size) != 0) {
      *native_errno = errno;
      return SAGR_STATUS_UNAVAILABLE;
    }
    if (socket_error == 0) {
      return SAGR_STATUS_SUCCESS;
    }
    if (socket_error == EINPROGRESS || socket_error == EALREADY) {
      continue;
    }
    *native_errno = socket_error;
    return map_connect_errno(socket_error);
  }
}

static sagr_status_t generate_handshake_identity(
    const monotonic_deadline_t *deadline, uint64_t *request_id,
    uint8_t nonce[16], int cancel_fd, int *native_errno) {
  uint8_t random_bytes[24];
  size_t offset = 0;
  while (offset < sizeof(random_bytes)) {
    ssize_t count;
    sagr_status_t deadline_status =
        check_operation_state(deadline, cancel_fd, native_errno);
    if (deadline_status != SAGR_STATUS_SUCCESS) {
      return deadline_status;
    }
    count = getrandom(random_bytes + offset, sizeof(random_bytes) - offset,
                      GRND_NONBLOCK);
    if (count > 0) {
      offset += (size_t)count;
      continue;
    }
    if (count < 0 && errno == EINTR) {
      continue;
    }
    *native_errno = count == 0 ? EIO : errno;
    return *native_errno == EAGAIN ? SAGR_STATUS_OUT_OF_RESOURCES
                                   : SAGR_STATUS_INTERNAL_ERROR;
  }
  *request_id = ((uint64_t)random_bytes[0] << 56) |
                ((uint64_t)random_bytes[1] << 48) |
                ((uint64_t)random_bytes[2] << 40) |
                ((uint64_t)random_bytes[3] << 32) |
                ((uint64_t)random_bytes[4] << 24) |
                ((uint64_t)random_bytes[5] << 16) |
                ((uint64_t)random_bytes[6] << 8) | random_bytes[7];
  memcpy(nonce, random_bytes + 8, 16);
  if (*request_id == 0) {
    *request_id = 1;
  }
  if (bytes_are_zero(nonce, 16)) {
    nonce[15] = 1;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t send_record(int socket_fd, const uint8_t *frame,
                                 size_t frame_size,
                                 const monotonic_deadline_t *deadline,
                                 int cancel_fd,
                                 int *native_errno, int *record_sent) {
  if (record_sent != NULL) {
    *record_sent = 0;
  }
  for (;;) {
    struct iovec vector;
    struct msghdr message;
    ssize_t count;
    sagr_status_t status =
        check_operation_state(deadline, cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
    memset(&message, 0, sizeof(message));
    vector.iov_base = (void *)frame;
    vector.iov_len = frame_size;
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    count = sendmsg(socket_fd, &message, MSG_NOSIGNAL);
    if (count == (ssize_t)frame_size) {
      if (record_sent != NULL) {
        *record_sent = 1;
      }
      return SAGR_STATUS_SUCCESS;
    }
    if (count >= 0) {
      if (record_sent != NULL) {
        *record_sent = 1;
      }
      *native_errno = EIO;
      return SAGR_STATUS_CONNECTION_LOST;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      status = wait_for_socket(socket_fd, POLLOUT, deadline, cancel_fd,
                               native_errno);
      if (status != SAGR_STATUS_SUCCESS) {
        return status;
      }
      continue;
    }
    *native_errno = errno;
    if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN) {
      return SAGR_STATUS_CONNECTION_LOST;
    }
    return SAGR_STATUS_UNAVAILABLE;
  }
}

static sagr_status_t send_record_with_descriptor(
    int socket_fd, const uint8_t *frame, size_t frame_size, int descriptor,
    const monotonic_deadline_t *deadline, int cancel_fd, int *native_errno,
    int *record_sent) {
  *record_sent = 0;
  for (;;) {
    struct iovec vector;
    struct msghdr message;
    struct cmsghdr *control_message;
    unsigned char control[CMSG_SPACE(sizeof(int))];
    ssize_t count;
    sagr_status_t status =
        check_operation_state(deadline, cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
    memset(&message, 0, sizeof(message));
    memset(control, 0, sizeof(control));
    vector.iov_base = (void *)frame;
    vector.iov_len = frame_size;
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    if (descriptor >= 0) {
      message.msg_control = control;
      message.msg_controllen = sizeof(control);
      control_message = CMSG_FIRSTHDR(&message);
      if (control_message == NULL) {
        *native_errno = EIO;
        return SAGR_STATUS_INTERNAL_ERROR;
      }
      control_message->cmsg_level = SOL_SOCKET;
      control_message->cmsg_type = SCM_RIGHTS;
      control_message->cmsg_len = CMSG_LEN(sizeof(descriptor));
      memcpy(CMSG_DATA(control_message), &descriptor, sizeof(descriptor));
    }
    count = sendmsg(socket_fd, &message, MSG_NOSIGNAL);
    if (count == (ssize_t)frame_size) {
      *record_sent = 1;
      return SAGR_STATUS_SUCCESS;
    }
    if (count >= 0) {
      *record_sent = 1;
      *native_errno = EIO;
      return SAGR_STATUS_CONNECTION_LOST;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      status = wait_for_socket(socket_fd, POLLOUT, deadline, cancel_fd,
                               native_errno);
      if (status != SAGR_STATUS_SUCCESS) {
        return status;
      }
      continue;
    }
    *native_errno = errno;
    if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN) {
      return SAGR_STATUS_CONNECTION_LOST;
    }
    return SAGR_STATUS_UNAVAILABLE;
  }
}

static int close_received_descriptors(struct msghdr *message) {
  struct cmsghdr *control_message;
  const int received_control = message->msg_controllen != 0;
  for (control_message = CMSG_FIRSTHDR(message); control_message != NULL;
       control_message = CMSG_NXTHDR(message, control_message)) {
    if (control_message->cmsg_level == SOL_SOCKET &&
        control_message->cmsg_type == SCM_RIGHTS &&
        control_message->cmsg_len >= CMSG_LEN(0)) {
      const size_t descriptor_bytes =
          control_message->cmsg_len - CMSG_LEN(0);
      const size_t descriptor_count = descriptor_bytes / sizeof(int);
      const int *descriptors = (const int *)CMSG_DATA(control_message);
      size_t descriptor_index;
      for (descriptor_index = 0; descriptor_index < descriptor_count;
           ++descriptor_index) {
        (void)close(descriptors[descriptor_index]);
      }
    }
  }
  return received_control;
}

static sagr_status_t receive_record(int socket_fd, uint8_t *frame,
                                    size_t frame_capacity, size_t *frame_size,
                                    const monotonic_deadline_t *deadline,
                                    int cancel_fd,
                                    int *native_errno) {
  for (;;) {
    struct iovec vector;
    struct msghdr message;
    unsigned char control[CMSG_SPACE(sizeof(int) * 4U)];
    ssize_t count;
    sagr_status_t status =
        check_operation_state(deadline, cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
    memset(&message, 0, sizeof(message));
    memset(control, 0, sizeof(control));
    vector.iov_base = frame;
    vector.iov_len = frame_capacity;
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    count = recvmsg(socket_fd, &message, MSG_CMSG_CLOEXEC);
    if (count >= 0) {
      const int received_control = close_received_descriptors(&message);
      if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      if (received_control != 0) {
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
    }
    if (count > 0) {
      *frame_size = (size_t)count;
      return SAGR_STATUS_SUCCESS;
    }
    if (count == 0) {
      return SAGR_STATUS_CONNECTION_LOST;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      status = wait_for_socket(socket_fd, POLLIN, deadline, cancel_fd,
                               native_errno);
      if (status != SAGR_STATUS_SUCCESS) {
        return status;
      }
      continue;
    }
    *native_errno = errno;
    if (errno == ECONNRESET || errno == ENOTCONN) {
      return SAGR_STATUS_CONNECTION_LOST;
    }
    return SAGR_STATUS_UNAVAILABLE;
  }
}

static int queue_capability_selected(const struct sagr_instance *instance) {
  return (instance->info
              .negotiated_capabilities[SAGR_CAPABILITY_QUEUE_WORD] &
          SAGR_CAPABILITY_QUEUE_MASK) != 0;
}

static int memory_capability_selected(const struct sagr_instance *instance) {
  return (instance->info
              .negotiated_capabilities[SAGR_CAPABILITY_MEMORY_WORD] &
          SAGR_CAPABILITY_MEMORY_MASK) != 0;
}

static int signal_capability_selected(const struct sagr_instance *instance) {
  return (instance->info
              .negotiated_capabilities[SAGR_CAPABILITY_SIGNAL_WORD] &
          SAGR_CAPABILITY_SIGNAL_MASK) != 0;
}

static int dispatch_capability_selected(const struct sagr_instance *instance) {
  return (instance->info
              .negotiated_capabilities[SAGR_CAPABILITY_DISPATCH_WORD] &
          SAGR_CAPABILITY_DISPATCH_MASK) != 0;
}

static int kmt_capability_selected(const struct sagr_instance *instance) {
  return (instance->info.negotiated_capabilities[SAGR_KMT_CAPABILITY_WORD] &
          SAGR_KMT_CAPABILITY_MASK) != 0;
}

static int code_object_capability_selected(
    const struct sagr_instance *instance) {
  return (instance->info.negotiated_capabilities[
              SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_WORD] &
          SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK) != 0;
}

static void poison_queue_transport(struct sagr_instance *instance) {
  instance->transport_poisoned = 1;
  if (instance->socket_fd >= 0) {
    (void)close(instance->socket_fd);
    instance->socket_fd = -1;
  }
}

static sagr_status_t require_queue_transport(
    const struct sagr_instance *instance, sagr_error_info_t *error,
    uint32_t error_size) {
  if (instance->transport_poisoned != 0 || instance->socket_fd < 0) {
    return fail_open(error, error_size, SAGR_STATUS_CONNECTION_LOST, -1, 0,
                     "queue transport is no longer reusable");
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t require_memory_transport(
    const struct sagr_instance *instance, sagr_error_info_t *error,
    uint32_t error_size) {
  if (instance->transport_poisoned != 0 || instance->socket_fd < 0) {
    return fail_open(error, error_size, SAGR_STATUS_CONNECTION_LOST, -1, 0,
                     "memory transport is no longer reusable");
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t require_signal_transport(
    const struct sagr_instance *instance, sagr_error_info_t *error,
    uint32_t error_size) {
  if (instance->transport_poisoned != 0 || instance->socket_fd < 0) {
    return fail_open(error, error_size, SAGR_STATUS_CONNECTION_LOST, -1, 0,
                     "signal transport is no longer reusable");
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t require_dispatch_transport(
    const struct sagr_instance *instance, sagr_error_info_t *error,
    uint32_t error_size) {
  if (instance->transport_poisoned != 0 || instance->socket_fd < 0) {
    return fail_open(error, error_size, SAGR_STATUS_CONNECTION_LOST, -1, 0,
                     "dispatch transport is no longer reusable");
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t require_code_object_transport(
    const struct sagr_instance *instance, sagr_error_info_t *error,
    uint32_t error_size) {
  if (instance->transport_poisoned != 0 || instance->socket_fd < 0) {
    return fail_open(error, error_size, SAGR_STATUS_CONNECTION_LOST, -1, 0,
                     "code-object transport is no longer reusable");
  }
  return SAGR_STATUS_SUCCESS;
}

static int code_object_commit_matches_manifest(
    const sagr_wire_code_object_request_t *request, uint64_t image_size,
    uint32_t chunk_count) {
  return request != NULL && request->opcode ==
                                SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT &&
         request->image_offset == 0U &&
         request->byte_count == image_size &&
         request->chunk_index == chunk_count && request->chunk_crc32c == 0U;
}

static int memory_id_is_active(const struct sagr_instance *instance,
                               uint64_t allocation_id) {
  const struct sagr_memory *memory;
  for (memory = instance->memories; memory != NULL; memory = memory->next) {
    if (memory->magic == SAGR_MEMORY_MAGIC &&
        memory->allocation_id == allocation_id) {
      return 1;
    }
  }
  return 0;
}

static int memory_range_overlaps(const struct sagr_instance *instance,
                                 uint64_t base, uint64_t size) {
  const struct sagr_memory *memory;
  const uint64_t end = base + size;
  for (memory = instance->memories; memory != NULL; memory = memory->next) {
    const uint64_t other_end = memory->simulated_va + memory->size_bytes;
    if (base < other_end && memory->simulated_va < end) {
      return 1;
    }
  }
  return 0;
}

static uint64_t signal_value_to_bits(int64_t value) {
  uint64_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int64_t signal_bits_to_value(uint64_t bits) {
  int64_t value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static int signal_predicate_satisfied(uint64_t condition,
                                      uint64_t value_bits,
                                      uint64_t compare_bits) {
  const int64_t value = signal_bits_to_value(value_bits);
  const int64_t compare = signal_bits_to_value(compare_bits);
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

static struct sagr_signal *find_signal(struct sagr_instance *instance,
                                       uint64_t signal_id,
                                       uint64_t generation) {
  struct sagr_signal *signal;
  for (signal = instance->signals; signal != NULL; signal = signal->next) {
    if (signal->magic == SAGR_SIGNAL_MAGIC &&
        signal->signal_id == signal_id && signal->generation == generation) {
      return signal;
    }
  }
  return NULL;
}

static int signal_id_is_active(const struct sagr_instance *instance,
                               uint64_t signal_id) {
  const struct sagr_signal *signal;
  for (signal = instance->signals; signal != NULL; signal = signal->next) {
    if (signal->magic == SAGR_SIGNAL_MAGIC &&
        signal->signal_id == signal_id) {
      return 1;
    }
  }
  return 0;
}

static int dispatch_owns_signal(const struct sagr_instance *instance,
                                uint64_t signal_id,
                                uint64_t generation) {
  const struct sagr_pending_dispatch *pending = instance->pending_dispatch;
  return pending != NULL && pending->signal != NULL &&
         pending->signal->magic == SAGR_SIGNAL_MAGIC &&
         pending->ack.signal_id == signal_id &&
         pending->ack.signal_generation == generation;
}

static int signal_completion_is_canonical(
    const struct sagr_signal *signal,
    const sagr_wire_signal_response_t *completion,
    int allow_untriggered) {
  if (!signal->wait_pending ||
      completion->message_type != SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION ||
      completion->status != SAGR_WIRE_STATUS_OK ||
      completion->opcode != SAGR_WIRE_SIGNAL_OPCODE_WAIT ||
      completion->signal_id != signal->signal_id ||
      completion->generation != signal->generation ||
      completion->sequence != signal->wait_sequence ||
      completion->request_id != signal->wait_request_id ||
      completion->ready != 0 || completion->sim_tick <= signal->wait_ack_tick ||
      !signal_predicate_satisfied(signal->wait_condition,
                                  completion->value_bits,
                                  signal->wait_compare_bits)) {
    return 0;
  }
  if (signal->wait_ready != 0) {
    return signal->wait_ack_tick != UINT64_MAX &&
           completion->sim_tick == signal->wait_ack_tick + UINT64_C(1) &&
           completion->value_bits == signal->wait_ack_value_bits;
  }
  if (signal->wait_trigger_known != 0) {
    return signal->wait_trigger_tick != UINT64_MAX &&
           completion->sim_tick == signal->wait_trigger_tick + UINT64_C(1) &&
           completion->value_bits == signal->wait_trigger_value_bits;
  }
  return allow_untriggered != 0 ||
         (dispatch_owns_signal(signal->instance, completion->signal_id,
                               completion->generation) &&
          completion->value_bits == UINT64_C(0));
}

static int record_dispatch_signal_completion(
    struct sagr_instance *instance,
    const sagr_wire_signal_response_t *completion) {
  struct sagr_pending_dispatch *pending = instance->pending_dispatch;
  if (pending == NULL || pending->signal == NULL ||
      pending->signal->magic != SAGR_SIGNAL_MAGIC ||
      completion->signal_id != pending->ack.signal_id ||
      completion->generation != pending->ack.signal_generation) {
    return 1;
  }
  if (completion->value_bits != pending->expected_signal_value_bits ||
      (pending->signal_completion_seen != 0 &&
       (pending->signal_completion_value_bits != completion->value_bits ||
        pending->signal_completion_tick != completion->sim_tick))) {
    return 0;
  }
  pending->signal_completion_value_bits = completion->value_bits;
  pending->signal_completion_tick = completion->sim_tick;
  pending->signal_completion_seen = 1;
  pending->signal->value = signal_bits_to_value(completion->value_bits);
  return 1;
}

static sagr_status_t buffer_signal_completion(
    struct sagr_instance *instance,
    const sagr_wire_signal_response_t *completion,
    int allow_untriggered) {
  struct sagr_signal *signal =
      find_signal(instance, completion->signal_id, completion->generation);
  if (signal == NULL || signal->completion_buffered != 0 ||
      !signal_completion_is_canonical(signal, completion,
                                      allow_untriggered) ||
      !record_dispatch_signal_completion(instance, completion)) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  signal->buffered_completion = *completion;
  signal->completion_buffered = 1;
  return SAGR_STATUS_SUCCESS;
}

static int take_buffered_signal_completion(
    struct sagr_signal *signal, sagr_wire_signal_response_t *completion) {
  if (signal->completion_buffered == 0) {
    return 0;
  }
  if (!signal_completion_is_canonical(signal, &signal->buffered_completion,
                                      0)) {
    return -1;
  }
  *completion = signal->buffered_completion;
  memset(&signal->buffered_completion, 0,
         sizeof(signal->buffered_completion));
  signal->completion_buffered = 0;
  return 1;
}

static sagr_status_t decode_and_buffer_signal_completion(
    struct sagr_instance *instance, const uint8_t *frame, size_t frame_size,
    uint64_t active_request_id, uint64_t untriggered_signal_id,
    uint64_t untriggered_generation, const char **reason) {
  sagr_wire_signal_response_t completion;
  int32_t wire_status = -1;
  sagr_status_t status = sagr_protocol_decode_signal_response(
      frame, frame_size, &instance->info, 0,
      SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION, &completion, &wire_status, reason);
  if (status != SAGR_STATUS_SUCCESS ||
      (active_request_id != 0 &&
       completion.request_id == active_request_id)) {
    *reason = "invalid signal completion while awaiting another record";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  status = buffer_signal_completion(
      instance, &completion,
      untriggered_signal_id != 0 &&
          completion.signal_id == untriggered_signal_id &&
          completion.generation == untriggered_generation);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "could not buffer signal completion";
  }
  return status;
}

static uint16_t queue_frame_type(const uint8_t *frame, size_t frame_size) {
  if (frame == NULL || frame_size < 16) {
    return 0;
  }
  return (uint16_t)(((uint16_t)frame[14] << 8) | frame[15]);
}

static struct sagr_queue *find_queue(struct sagr_instance *instance,
                                     uint64_t queue_id,
                                     uint64_t generation) {
  struct sagr_queue *queue;
  for (queue = instance->queues; queue != NULL; queue = queue->next) {
    if (queue->magic == SAGR_QUEUE_MAGIC && queue->queue_id == queue_id &&
        queue->generation == generation) {
      return queue;
    }
  }
  return NULL;
}

static int dispatch_response_identities_match(
    const sagr_wire_dispatch_response_t *left,
    const sagr_wire_dispatch_response_t *right) {
  return left->queue_id == right->queue_id &&
         left->queue_generation == right->queue_generation &&
         left->queue_sequence == right->queue_sequence &&
         left->fixture_id == right->fixture_id &&
         left->input_allocation_id == right->input_allocation_id &&
         left->input_generation == right->input_generation &&
         left->output_allocation_id == right->output_allocation_id &&
         left->output_generation == right->output_generation &&
         left->signal_id == right->signal_id &&
         left->signal_generation == right->signal_generation;
}

static int dispatch_completion_is_canonical(
    const struct sagr_pending_dispatch *pending,
    const sagr_wire_dispatch_response_t *completion) {
  const sagr_wire_dispatch_response_t *ack;
  if (pending == NULL || completion == NULL) {
    return 0;
  }
  ack = &pending->ack;
  if (completion->message_type != SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION ||
      completion->status != SAGR_WIRE_STATUS_OK ||
      completion->opcode != SAGR_WIRE_DISPATCH_OPCODE_SUBMIT_PINNED ||
      completion->request_id != pending->request_id ||
      !dispatch_response_identities_match(completion, ack) ||
      completion->trace_id != ack->trace_id ||
      completion->input_gpu_va != ack->input_gpu_va ||
      completion->output_gpu_va != ack->output_gpu_va ||
      completion->packet_crc32c != ack->packet_crc32c ||
      completion->output_crc32c != SAGR_DISPATCH_OUTPUT_CRC32C ||
      completion->admission_tick != ack->admission_tick ||
      completion->admission_tick >= completion->start_tick ||
      completion->start_tick > completion->end_tick ||
      completion->end_tick > completion->retire_tick ||
      pending->signal_completion_seen == 0 ||
      pending->signal_completion_value_bits !=
          pending->expected_signal_value_bits ||
      completion->retire_tick == UINT64_MAX ||
      pending->signal_completion_tick !=
          completion->retire_tick + UINT64_C(1) ||
      completion->admission_tick > UINT64_MAX - UINT64_C(2) ||
      completion->retire_tick <= completion->admission_tick + UINT64_C(1)) {
    return 0;
  }
  return 1;
}

static void observe_dispatch_signal_mirror(
    struct sagr_pending_dispatch *pending) {
  if (pending->signal != NULL &&
      pending->signal->magic == SAGR_SIGNAL_MAGIC &&
      pending->signal->signal_id == pending->ack.signal_id &&
      pending->signal->generation == pending->ack.signal_generation) {
    pending->signal->value = SAGR_DISPATCH_EXPECTED_SIGNAL_VALUE;
  }
}

static sagr_status_t buffer_dispatch_completion(
    struct sagr_instance *instance,
    const sagr_wire_dispatch_response_t *completion) {
  struct sagr_pending_dispatch *pending = instance->pending_dispatch;
  if (pending == NULL || pending->completion_buffered != 0 ||
      pending->dispatch_completion_consumed != 0 ||
      !dispatch_completion_is_canonical(pending, completion)) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  pending->buffered_completion = *completion;
  pending->completion_buffered = 1;
  observe_dispatch_signal_mirror(pending);
  return SAGR_STATUS_SUCCESS;
}

static int take_buffered_dispatch_completion(
    struct sagr_pending_dispatch *pending,
    sagr_wire_dispatch_response_t *completion) {
  if (pending->completion_buffered == 0) {
    return 0;
  }
  if (!dispatch_completion_is_canonical(pending,
                                        &pending->buffered_completion)) {
    return -1;
  }
  *completion = pending->buffered_completion;
  memset(&pending->buffered_completion, 0,
         sizeof(pending->buffered_completion));
  pending->completion_buffered = 0;
  observe_dispatch_signal_mirror(pending);
  return 1;
}

static sagr_status_t decode_and_buffer_dispatch_completion(
    struct sagr_instance *instance, const uint8_t *frame, size_t frame_size,
    uint64_t active_request_id, const char **reason) {
  sagr_wire_dispatch_response_t completion;
  int32_t wire_status = -1;
  sagr_status_t status = sagr_protocol_decode_dispatch_response(
      frame, frame_size, &instance->info, 0,
      SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION, &completion, &wire_status,
      reason);
  if (status != SAGR_STATUS_SUCCESS || wire_status != SAGR_WIRE_STATUS_OK ||
      (active_request_id != 0 && completion.request_id == active_request_id)) {
    *reason = "invalid dispatch completion while awaiting another record";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  status = buffer_dispatch_completion(instance, &completion);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "could not buffer dispatch completion";
  }
  return status;
}

static void release_consumed_dispatch(struct sagr_instance *instance) {
  struct sagr_pending_dispatch *pending = instance->pending_dispatch;
  if (pending != NULL && pending->dispatch_completion_consumed != 0 &&
      pending->signal_completion_consumed != 0) {
    memset(pending, 0, sizeof(*pending));
    free(pending);
    instance->pending_dispatch = NULL;
  }
}

static int queue_id_is_active(const struct sagr_instance *instance,
                              uint64_t queue_id) {
  const struct sagr_queue *queue;
  for (queue = instance->queues; queue != NULL; queue = queue->next) {
    if (queue->magic == SAGR_QUEUE_MAGIC && queue->queue_id == queue_id) {
      return 1;
    }
  }
  return 0;
}

static void queue_remove_pending(struct sagr_queue *queue, uint64_t sequence) {
  uint32_t index;
  for (index = 0; index < queue->pending_count; ++index) {
    if (queue->pending_sequences[index] == sequence) {
      if (index + 1U < queue->pending_count) {
        memmove(&queue->pending_sequences[index],
                &queue->pending_sequences[index + 1U],
                (size_t)(queue->pending_count - index - 1U) *
                    sizeof(queue->pending_sequences[0]));
        memmove(&queue->pending_kinds[index], &queue->pending_kinds[index + 1U],
                (size_t)(queue->pending_count - index - 1U) *
                    sizeof(queue->pending_kinds[0]));
        memmove(&queue->pending_request_ids[index],
                &queue->pending_request_ids[index + 1U],
                (size_t)(queue->pending_count - index - 1U) *
                    sizeof(queue->pending_request_ids[0]));
        memmove(&queue->pending_ack_ticks[index],
                &queue->pending_ack_ticks[index + 1U],
                (size_t)(queue->pending_count - index - 1U) *
                    sizeof(queue->pending_ack_ticks[0]));
      }
      --queue->pending_count;
      return;
    }
  }
}

static int queue_pending_kind(const struct sagr_queue *queue,
                              uint64_t sequence, uint64_t *kind,
                              uint64_t *request_id, uint64_t *ack_tick) {
  uint32_t index;
  for (index = 0; index < queue->pending_count; ++index) {
    if (queue->pending_sequences[index] == sequence) {
      *kind = queue->pending_kinds[index];
      *request_id = queue->pending_request_ids[index];
      *ack_tick = queue->pending_ack_ticks[index];
      return 1;
    }
  }
  return 0;
}

static int queue_completion_is_canonical(
    const sagr_wire_queue_response_t *completion, uint64_t command_kind,
    uint64_t ack_tick) {
  if (completion->opcode != SAGR_WIRE_QUEUE_OPCODE_DOORBELL ||
      ack_tick == UINT64_MAX || completion->sim_tick != ack_tick + UINT64_C(1)) {
    return 0;
  }
  if (command_kind == SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST) {
    return completion->status == SAGR_WIRE_STATUS_INTERNAL &&
           completion->value == SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST &&
           completion->error_code == UINT64_C(1);
  }
  return (command_kind == SAGR_QUEUE_COMMAND_NOOP ||
          command_kind == SAGR_QUEUE_COMMAND_CONTROL_TEST) &&
         completion->status == SAGR_WIRE_STATUS_OK &&
         completion->value == command_kind && completion->error_code == 0;
}

static sagr_status_t buffer_completion(
    struct sagr_instance *instance,
    const sagr_wire_queue_response_t *completion) {
  struct sagr_queue *queue = find_queue(instance, completion->queue_id,
                                         completion->generation);
  uint32_t index;
  uint64_t expected_kind = 0;
  uint64_t expected_request_id = 0;
  uint64_t expected_ack_tick = 0;
  if (queue == NULL || completion->opcode != SAGR_WIRE_QUEUE_OPCODE_DOORBELL ||
      completion->sequence == 0) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (queue_pending_kind(queue, completion->sequence, &expected_kind,
                         &expected_request_id, &expected_ack_tick)) {
    if (completion->request_id != expected_request_id ||
        !queue_completion_is_canonical(completion, expected_kind,
                                       expected_ack_tick)) {
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
  } else {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  for (index = 0; index < queue->buffered_count; ++index) {
    if (queue->buffered[index].sequence == completion->sequence) {
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
  }
  if (queue->buffered_count >= SAGR_QUEUE_MAX_INFLIGHT) {
    return SAGR_STATUS_OUT_OF_RESOURCES;
  }
  queue->buffered[queue->buffered_count++] = *completion;
  return SAGR_STATUS_SUCCESS;
}

static int take_buffered_completion(
    struct sagr_queue *queue, uint64_t sequence, uint64_t request_id,
    sagr_wire_queue_response_t *completion) {
  uint32_t index;
  for (index = 0; index < queue->buffered_count; ++index) {
    if (queue->buffered[index].sequence == sequence &&
        queue->buffered[index].request_id == request_id) {
      *completion = queue->buffered[index];
      if (index + 1U < queue->buffered_count) {
        memmove(&queue->buffered[index], &queue->buffered[index + 1U],
                (size_t)(queue->buffered_count - index - 1U) *
                    sizeof(queue->buffered[0]));
      }
      --queue->buffered_count;
      return 1;
    }
  }
  return 0;
}

static sagr_status_t map_queue_wire_status(uint16_t opcode,
                                           int32_t wire_status) {
  if ((opcode == SAGR_WIRE_QUEUE_OPCODE_DESTROY ||
       opcode == SAGR_WIRE_QUEUE_OPCODE_DOORBELL) &&
      wire_status == SAGR_WIRE_STATUS_PROTOCOL_STATE) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (wire_status < 0) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  return sagr_protocol_map_wire_status((uint32_t)wire_status);
}

static sagr_status_t exchange_queue_request(
    struct sagr_instance *instance, const sagr_wire_queue_request_t *request,
    const monotonic_deadline_t *deadline, int cancel_fd,
    sagr_wire_queue_response_t *response, int32_t *wire_status,
    int *native_errno, const char **reason, uint64_t *out_request_id) {
  uint8_t request_frame[SAGR_WIRE_QUEUE_FRAME_BYTES];
  uint8_t response_frame[SAGR_WIRE_MAX_RECORD_BYTES];
  size_t request_size = 0;
  uint64_t request_id = 0;
  sagr_status_t status = sagr_protocol_allocate_request_id(
      &instance->next_request_id, &request_id);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "queue request ID space exhausted";
    return status;
  }
  status = sagr_protocol_encode_queue_request(
      &instance->info, request_id, request, request_frame,
      sizeof(request_frame), &request_size);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "could not encode queue request";
    return status;
  }
  if (out_request_id != NULL) {
    *out_request_id = request_id;
  }
  status = send_record(instance->socket_fd, request_frame, request_size,
                       deadline, cancel_fd, native_errno, NULL);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "queue request send failed";
    return status;
  }

  for (;;) {
    size_t response_size = 0;
    uint16_t message_type;
    int32_t decoded_wire_status = -1;
    sagr_wire_queue_response_t decoded;
    status = receive_record(instance->socket_fd, response_frame,
                            sizeof(response_frame), &response_size, deadline,
                            cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      *reason = "queue response receive failed";
      poison_queue_transport(instance);
      return status;
    }
    message_type = queue_frame_type(response_frame, response_size);
    if (message_type == SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) {
      status = decode_and_buffer_signal_completion(
          instance, response_frame, response_size, request_id, 0, 0, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) {
      status = sagr_protocol_decode_queue_response(
          response_frame, response_size, &instance->info, 0,
          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &decoded,
          &decoded_wire_status, reason);
      if (status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) {
        poison_queue_transport(instance);
        return status;
      }
      if (decoded.request_id == request_id) {
        *reason = "queue completion arrived before its ACK";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      status = buffer_completion(instance, &decoded);
      if (status != SAGR_STATUS_SUCCESS) {
        *reason = "could not buffer queue completion";
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) {
      status = decode_and_buffer_dispatch_completion(
          instance, response_frame, response_size, request_id, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    status = sagr_protocol_decode_queue_response(
        response_frame, response_size, &instance->info, request_id,
        SAGR_WIRE_MESSAGE_QUEUE_ACK, &decoded, &decoded_wire_status, reason);
    *wire_status = decoded_wire_status;
    if (status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) {
      poison_queue_transport(instance);
      return status;
    }
    if (decoded.opcode != request->opcode) {
      *reason = "queue ACK opcode mismatch";
      poison_queue_transport(instance);
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    *response = decoded;
    if (status != SAGR_STATUS_SUCCESS && decoded_wire_status >= 0) {
      if (sagr_protocol_validate_failed_queue_ack(request, &decoded) !=
          SAGR_STATUS_SUCCESS) {
        *reason = "noncanonical failed queue ACK";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      return map_queue_wire_status(request->opcode, decoded_wire_status);
    }
    return status;
  }
}

static sagr_status_t map_memory_wire_status(uint16_t opcode,
                                            int32_t wire_status) {
  if (opcode != SAGR_WIRE_MEMORY_OPCODE_ALLOC &&
      wire_status == SAGR_WIRE_STATUS_PROTOCOL_STATE) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (wire_status < 0) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  return sagr_protocol_map_wire_status((uint32_t)wire_status);
}

static sagr_status_t exchange_memory_request(
    struct sagr_instance *instance, const sagr_wire_memory_request_t *request,
    int staging_fd, const monotonic_deadline_t *deadline, int cancel_fd,
    sagr_wire_memory_response_t *response, int32_t *wire_status,
    int *native_errno, const char **reason) {
  uint8_t request_frame[SAGR_WIRE_MEMORY_FRAME_BYTES];
  uint8_t response_frame[SAGR_WIRE_MAX_RECORD_BYTES];
  size_t request_size = 0;
  uint64_t request_id = 0;
  int record_sent = 0;
  sagr_status_t status = sagr_protocol_allocate_request_id(
      &instance->next_request_id, &request_id);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "memory request ID space exhausted";
    return status;
  }
  status = sagr_protocol_encode_memory_request(
      &instance->info, request_id, request, request_frame,
      sizeof(request_frame), &request_size);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "could not encode memory request";
    return status;
  }
  status = send_record_with_descriptor(
      instance->socket_fd, request_frame, request_size, staging_fd, deadline,
      cancel_fd, native_errno, &record_sent);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "memory request send failed";
    if (record_sent != 0) {
      poison_queue_transport(instance);
    }
    return status;
  }

  for (;;) {
    size_t response_size = 0;
    uint16_t message_type;
    int32_t decoded_wire_status = -1;
    sagr_wire_memory_response_t decoded;
    status = receive_record(instance->socket_fd, response_frame,
                            sizeof(response_frame), &response_size, deadline,
                            cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      *reason = "memory ACK receive failed";
      poison_queue_transport(instance);
      return status;
    }
    message_type = queue_frame_type(response_frame, response_size);
    if (message_type == SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) {
      status = decode_and_buffer_signal_completion(
          instance, response_frame, response_size, request_id, 0, 0, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) {
      sagr_wire_queue_response_t completion;
      status = sagr_protocol_decode_queue_response(
          response_frame, response_size, &instance->info, 0,
          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &completion,
          &decoded_wire_status, reason);
      if (status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) {
        poison_queue_transport(instance);
        return status;
      }
      if (completion.request_id == request_id ||
          buffer_completion(instance, &completion) != SAGR_STATUS_SUCCESS) {
        *reason = "invalid queue completion while waiting for memory ACK";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) {
      status = decode_and_buffer_dispatch_completion(
          instance, response_frame, response_size, request_id, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    status = sagr_protocol_decode_memory_response(
        response_frame, response_size, &instance->info, request_id, &decoded,
        &decoded_wire_status, reason);
    *wire_status = decoded_wire_status;
    if (status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) {
      poison_queue_transport(instance);
      return status;
    }
    if (decoded.opcode != request->opcode) {
      *reason = "memory ACK opcode mismatch";
      poison_queue_transport(instance);
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    *response = decoded;
    if (status != SAGR_STATUS_SUCCESS) {
      if (sagr_protocol_validate_failed_memory_ack(request, &decoded) !=
          SAGR_STATUS_SUCCESS) {
        *reason = "noncanonical failed memory ACK";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      return map_memory_wire_status(request->opcode, decoded_wire_status);
    }
    return SAGR_STATUS_SUCCESS;
  }
}

static sagr_status_t receive_queue_completion(
    struct sagr_queue *queue, uint64_t sequence, uint64_t request_id,
    uint64_t command_kind, uint64_t ack_tick,
    const monotonic_deadline_t *deadline, int cancel_fd,
    sagr_wire_queue_response_t *completion, int32_t *wire_status,
    int *native_errno, const char **reason) {
  struct sagr_instance *instance = queue->instance;
  uint8_t frame[SAGR_WIRE_MAX_RECORD_BYTES];
  sagr_status_t state =
      check_operation_state(deadline, cancel_fd, native_errno);
  if (state != SAGR_STATUS_SUCCESS) {
    *reason = "queue completion wait was cancelled or expired";
    return state;
  }
  if (take_buffered_completion(queue, sequence, request_id, completion)) {
    *wire_status = (int32_t)completion->status;
    return map_queue_wire_status(completion->opcode, *wire_status);
  }
  for (;;) {
    sagr_wire_queue_response_t decoded;
    size_t frame_size = 0;
    int32_t decoded_wire_status = -1;
    sagr_status_t status = receive_record(
        instance->socket_fd, frame, sizeof(frame), &frame_size, deadline,
        cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      *reason = "queue completion receive failed";
      if (status != SAGR_STATUS_TIMED_OUT && status != SAGR_STATUS_CANCELLED) {
        poison_queue_transport(instance);
      }
      return status;
    }
    if (queue_frame_type(frame, frame_size) ==
        SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) {
      status = decode_and_buffer_signal_completion(
          instance, frame, frame_size, 0, 0, 0, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    if (queue_frame_type(frame, frame_size) ==
        SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) {
      status = decode_and_buffer_dispatch_completion(
          instance, frame, frame_size, 0, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    if (queue_frame_type(frame, frame_size) !=
        SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) {
      *reason = "unexpected record while waiting for queue completion";
      poison_queue_transport(instance);
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    status = sagr_protocol_decode_queue_response(
        frame, frame_size, &instance->info, 0,
        SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &decoded, &decoded_wire_status,
        reason);
    if (status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) {
      poison_queue_transport(instance);
      return status;
    }
    if (decoded.queue_id == queue->queue_id &&
        decoded.generation == queue->generation &&
        decoded.sequence == sequence) {
      if (decoded.request_id != request_id) {
        *reason = "queue completion request ID mismatch";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      if (!queue_completion_is_canonical(&decoded, command_kind, ack_tick)) {
        *reason = "noncanonical queue completion";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      *completion = decoded;
      *wire_status = decoded_wire_status;
      return map_queue_wire_status(decoded.opcode, decoded_wire_status);
    }
    status = buffer_completion(instance, &decoded);
    if (status != SAGR_STATUS_SUCCESS) {
      *reason = "could not buffer out-of-order queue completion";
      poison_queue_transport(instance);
      return status;
    }
  }
}

static sagr_status_t map_signal_wire_status(uint16_t opcode,
                                            int32_t wire_status) {
  if (opcode != SAGR_WIRE_SIGNAL_OPCODE_CREATE &&
      wire_status == SAGR_WIRE_STATUS_PROTOCOL_STATE) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (wire_status < 0) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  return sagr_protocol_map_wire_status((uint32_t)wire_status);
}

static sagr_status_t exchange_signal_request(
    struct sagr_instance *instance, const sagr_wire_signal_request_t *request,
    const monotonic_deadline_t *deadline, int cancel_fd,
    sagr_wire_signal_response_t *response, int32_t *wire_status,
    int *native_errno, const char **reason, uint64_t *out_request_id) {
  uint8_t request_frame[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  uint8_t response_frame[SAGR_WIRE_MAX_RECORD_BYTES];
  size_t request_size = 0;
  uint64_t request_id = 0;
  int record_sent = 0;
  sagr_status_t status = sagr_protocol_allocate_request_id(
      &instance->next_request_id, &request_id);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "signal request ID space exhausted";
    return status;
  }
  status = sagr_protocol_encode_signal_request(
      &instance->info, request_id, request, request_frame,
      sizeof(request_frame), &request_size);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "could not encode signal request";
    return status;
  }
  if (out_request_id != NULL) {
    *out_request_id = request_id;
  }
  status = send_record(instance->socket_fd, request_frame, request_size,
                       deadline, cancel_fd, native_errno, &record_sent);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "signal request send failed";
    if (record_sent != 0) {
      poison_queue_transport(instance);
    }
    return status;
  }

  for (;;) {
    size_t response_size = 0;
    uint16_t message_type;
    int32_t decoded_wire_status = -1;
    sagr_wire_signal_response_t decoded;
    status = receive_record(instance->socket_fd, response_frame,
                            sizeof(response_frame), &response_size, deadline,
                            cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      *reason = "signal ACK receive failed";
      poison_queue_transport(instance);
      return status;
    }
    message_type = queue_frame_type(response_frame, response_size);
    if (message_type == SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) {
      sagr_wire_queue_response_t completion;
      status = sagr_protocol_decode_queue_response(
          response_frame, response_size, &instance->info, 0,
          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &completion,
          &decoded_wire_status, reason);
      if ((status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) ||
          completion.request_id == request_id ||
          buffer_completion(instance, &completion) != SAGR_STATUS_SUCCESS) {
        *reason = "invalid queue completion while waiting for signal ACK";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) {
      status = decode_and_buffer_signal_completion(
          instance, response_frame, response_size, request_id,
          request->opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE
              ? request->signal_id
              : 0,
          request->opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE
              ? request->generation
              : 0,
          reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) {
      status = decode_and_buffer_dispatch_completion(
          instance, response_frame, response_size, request_id, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    status = sagr_protocol_decode_signal_response(
        response_frame, response_size, &instance->info, request_id,
        SAGR_WIRE_MESSAGE_SIGNAL_ACK, &decoded, &decoded_wire_status, reason);
    *wire_status = decoded_wire_status;
    if (status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) {
      poison_queue_transport(instance);
      return status;
    }
    if (decoded.opcode != request->opcode) {
      *reason = "signal ACK opcode mismatch";
      poison_queue_transport(instance);
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    *response = decoded;
    if (status != SAGR_STATUS_SUCCESS) {
      if (sagr_protocol_validate_failed_signal_ack(request, &decoded) !=
          SAGR_STATUS_SUCCESS) {
        *reason = "noncanonical failed signal ACK";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      return map_signal_wire_status(request->opcode, decoded_wire_status);
    }
    return SAGR_STATUS_SUCCESS;
  }
}

static sagr_status_t map_dispatch_wire_status(int32_t wire_status) {
  if (wire_status == SAGR_WIRE_STATUS_PROTOCOL_STATE) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (wire_status < 0) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  return sagr_protocol_map_wire_status((uint32_t)wire_status);
}

static sagr_status_t exchange_dispatch_request(
    struct sagr_instance *instance,
    const sagr_wire_dispatch_request_t *request,
    const monotonic_deadline_t *deadline, int cancel_fd,
    sagr_wire_dispatch_response_t *response, int32_t *wire_status,
    int *native_errno, const char **reason, uint64_t *out_request_id) {
  uint8_t request_frame[SAGR_WIRE_DISPATCH_REQUEST_FRAME_BYTES];
  uint8_t response_frame[SAGR_WIRE_MAX_RECORD_BYTES];
  size_t request_size = 0;
  uint64_t request_id = 0;
  int record_sent = 0;
  sagr_status_t status = sagr_protocol_allocate_request_id(
      &instance->next_request_id, &request_id);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "dispatch request ID space exhausted";
    return status;
  }
  status = sagr_protocol_encode_dispatch_request(
      &instance->info, request_id, request, request_frame,
      sizeof(request_frame), &request_size);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "could not encode dispatch request";
    return status;
  }
  if (out_request_id != NULL) {
    *out_request_id = request_id;
  }
  status = send_record(instance->socket_fd, request_frame, request_size,
                       deadline, cancel_fd, native_errno, &record_sent);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = "dispatch request send failed";
    if (record_sent != 0) {
      poison_queue_transport(instance);
    }
    return status;
  }

  for (;;) {
    size_t response_size = 0;
    uint16_t message_type;
    int32_t decoded_wire_status = -1;
    sagr_wire_dispatch_response_t decoded;
    status = receive_record(instance->socket_fd, response_frame,
                            sizeof(response_frame), &response_size, deadline,
                            cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      *reason = "dispatch ACK receive failed";
      poison_queue_transport(instance);
      return status;
    }
    message_type = queue_frame_type(response_frame, response_size);
    if (message_type == SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) {
      sagr_wire_queue_response_t completion;
      status = sagr_protocol_decode_queue_response(
          response_frame, response_size, &instance->info, 0,
          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &completion,
          &decoded_wire_status, reason);
      if ((status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) ||
          completion.request_id == request_id ||
          buffer_completion(instance, &completion) != SAGR_STATUS_SUCCESS) {
        *reason = "invalid queue completion while waiting for dispatch ACK";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) {
      status = decode_and_buffer_signal_completion(
          instance, response_frame, response_size, request_id, 0, 0, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) {
      status = decode_and_buffer_dispatch_completion(
          instance, response_frame, response_size, request_id, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    status = sagr_protocol_decode_dispatch_response(
        response_frame, response_size, &instance->info, request_id,
        SAGR_WIRE_MESSAGE_DISPATCH_ACK, &decoded, &decoded_wire_status,
        reason);
    *wire_status = decoded_wire_status;
    if (status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) {
      poison_queue_transport(instance);
      return status;
    }
    if (decoded.opcode != request->opcode) {
      *reason = "dispatch ACK opcode mismatch";
      poison_queue_transport(instance);
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    *response = decoded;
    if (status != SAGR_STATUS_SUCCESS) {
      if (sagr_protocol_validate_failed_dispatch_ack(request, &decoded) !=
          SAGR_STATUS_SUCCESS) {
        *reason = "noncanonical failed dispatch ACK";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      return map_dispatch_wire_status(decoded_wire_status);
    }
    return SAGR_STATUS_SUCCESS;
  }
}

static sagr_status_t receive_dispatch_completion(
    struct sagr_pending_dispatch *pending,
    const monotonic_deadline_t *deadline, int cancel_fd,
    sagr_wire_dispatch_response_t *completion, int32_t *wire_status,
    int *native_errno, const char **reason) {
  struct sagr_instance *instance = pending->queue->instance;
  uint8_t frame[SAGR_WIRE_MAX_RECORD_BYTES];
  sagr_status_t state = check_operation_state(deadline, cancel_fd,
                                               native_errno);
  int buffered;
  if (state != SAGR_STATUS_SUCCESS) {
    *reason = "dispatch completion wait was cancelled or expired";
    return state;
  }
  buffered = take_buffered_dispatch_completion(pending, completion);
  if (buffered < 0) {
    *reason = "buffered dispatch completion is noncanonical";
    poison_queue_transport(instance);
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (buffered > 0) {
    *wire_status = (int32_t)completion->status;
    return SAGR_STATUS_SUCCESS;
  }
  for (;;) {
    size_t frame_size = 0;
    uint16_t message_type;
    int32_t decoded_wire_status = -1;
    sagr_status_t status = receive_record(
        instance->socket_fd, frame, sizeof(frame), &frame_size, deadline,
        cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      *reason = "dispatch completion receive failed";
      if (status != SAGR_STATUS_TIMED_OUT && status != SAGR_STATUS_CANCELLED) {
        poison_queue_transport(instance);
      }
      return status;
    }
    message_type = queue_frame_type(frame, frame_size);
    if (message_type == SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) {
      sagr_wire_queue_response_t queue_completion;
      status = sagr_protocol_decode_queue_response(
          frame, frame_size, &instance->info, 0,
          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &queue_completion,
          &decoded_wire_status, reason);
      if ((status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) ||
          buffer_completion(instance, &queue_completion) !=
              SAGR_STATUS_SUCCESS) {
        *reason = "invalid queue completion while waiting for dispatch";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) {
      status = decode_and_buffer_signal_completion(
          instance, frame, frame_size, 0, 0, 0, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    if (message_type != SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) {
      *reason = "unexpected record while waiting for dispatch completion";
      poison_queue_transport(instance);
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    {
      sagr_wire_dispatch_response_t decoded;
      status = sagr_protocol_decode_dispatch_response(
          frame, frame_size, &instance->info, 0,
          SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION, &decoded,
          &decoded_wire_status, reason);
      if (status != SAGR_STATUS_SUCCESS ||
          decoded_wire_status != SAGR_WIRE_STATUS_OK ||
          !dispatch_completion_is_canonical(pending, &decoded)) {
        *reason = "noncanonical dispatch completion";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      *completion = decoded;
      *wire_status = decoded_wire_status;
      observe_dispatch_signal_mirror(pending);
      return SAGR_STATUS_SUCCESS;
    }
  }
}

static sagr_status_t receive_signal_completion(
    struct sagr_signal *signal, const monotonic_deadline_t *deadline,
    int cancel_fd, sagr_wire_signal_response_t *completion,
    int32_t *wire_status, int *native_errno, const char **reason) {
  struct sagr_instance *instance = signal->instance;
  uint8_t frame[SAGR_WIRE_MAX_RECORD_BYTES];
  sagr_status_t state = check_operation_state(deadline, cancel_fd,
                                               native_errno);
  int buffered;
  if (state != SAGR_STATUS_SUCCESS) {
    *reason = "signal completion wait was cancelled or expired";
    return state;
  }
  buffered = take_buffered_signal_completion(signal, completion);
  if (buffered < 0) {
    *reason = "buffered signal completion is noncanonical";
    poison_queue_transport(instance);
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (buffered > 0) {
    *wire_status = (int32_t)completion->status;
    return SAGR_STATUS_SUCCESS;
  }
  for (;;) {
    size_t frame_size = 0;
    uint16_t message_type;
    int32_t decoded_wire_status = -1;
    sagr_status_t status = receive_record(
        instance->socket_fd, frame, sizeof(frame), &frame_size, deadline,
        cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      *reason = "signal completion receive failed";
      if (status != SAGR_STATUS_TIMED_OUT && status != SAGR_STATUS_CANCELLED) {
        poison_queue_transport(instance);
      }
      return status;
    }
    message_type = queue_frame_type(frame, frame_size);
    if (message_type == SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) {
      sagr_wire_queue_response_t queue_completion;
      status = sagr_protocol_decode_queue_response(
          frame, frame_size, &instance->info, 0,
          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &queue_completion,
          &decoded_wire_status, reason);
      if ((status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) ||
          buffer_completion(instance, &queue_completion) !=
              SAGR_STATUS_SUCCESS) {
        *reason = "invalid queue completion while waiting for signal";
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) {
      status = decode_and_buffer_dispatch_completion(
          instance, frame, frame_size, 0, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return status;
      }
      continue;
    }
    if (message_type != SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) {
      *reason = "unexpected record while waiting for signal completion";
      poison_queue_transport(instance);
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    {
      sagr_wire_signal_response_t decoded;
      status = sagr_protocol_decode_signal_response(
          frame, frame_size, &instance->info, 0,
          SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION, &decoded,
          &decoded_wire_status, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        poison_queue_transport(instance);
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      if (decoded.signal_id == signal->signal_id &&
          decoded.generation == signal->generation &&
          decoded.sequence == signal->wait_sequence) {
        if (!signal_completion_is_canonical(signal, &decoded, 0)) {
          *reason = "noncanonical signal completion";
          poison_queue_transport(instance);
          return SAGR_STATUS_PROTOCOL_ERROR;
        }
        if (!record_dispatch_signal_completion(instance, &decoded)) {
          *reason = "dispatch signal completion mismatch";
          poison_queue_transport(instance);
          return SAGR_STATUS_PROTOCOL_ERROR;
        }
        *completion = decoded;
        *wire_status = decoded_wire_status;
        return SAGR_STATUS_SUCCESS;
      }
      status = buffer_signal_completion(instance, &decoded, 0);
      if (status != SAGR_STATUS_SUCCESS) {
        *reason = "could not buffer out-of-order signal completion";
        poison_queue_transport(instance);
        return status;
      }
    }
  }
}

static sagr_status_t receive_code_object_ack(
    struct sagr_instance *instance,
    const sagr_wire_code_object_request_t *request, uint64_t request_id,
    const monotonic_deadline_t *deadline, int cancel_fd,
    sagr_wire_code_object_response_t *response, int32_t *wire_status,
    int *native_errno, const char **reason, int *poison) {
  uint8_t frame[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  for (;;) {
    size_t frame_size = 0U;
    uint16_t message_type;
    int32_t decoded_wire_status = -1;
    sagr_status_t status = receive_record(
        instance->socket_fd, frame, sizeof(frame), &frame_size, deadline,
        cancel_fd, native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      *reason = "code-object ACK receive failed";
      *poison = 1;
      return status;
    }
    message_type = queue_frame_type(frame, frame_size);
    if (message_type == SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) {
      sagr_wire_queue_response_t completion;
      status = sagr_protocol_decode_queue_response(
          frame, frame_size, &instance->info, 0,
          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &completion,
          &decoded_wire_status, reason);
      if ((status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) ||
          completion.request_id == request_id ||
          buffer_completion(instance, &completion) != SAGR_STATUS_SUCCESS) {
        *reason =
            "invalid queue completion while waiting for code-object ACK";
        *poison = 1;
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) {
      status = decode_and_buffer_signal_completion(
          instance, frame, frame_size, request_id, 0, 0, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        *poison = 1;
        return status;
      }
      continue;
    }
    if (message_type == SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) {
      status = decode_and_buffer_dispatch_completion(
          instance, frame, frame_size, request_id, reason);
      if (status != SAGR_STATUS_SUCCESS) {
        *poison = 1;
        return status;
      }
      continue;
    }
    status = sagr_protocol_decode_code_object_response(
        frame, frame_size, &instance->info, request_id, response,
        &decoded_wire_status, reason);
    *wire_status = decoded_wire_status;
    if (status != SAGR_STATUS_SUCCESS && decoded_wire_status < 0) {
      *poison = 1;
      return status;
    }
    if (response->opcode != request->opcode) {
      *reason = "code-object ACK opcode mismatch";
      *poison = 1;
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    return status;
  }
}

sagr_status_t sagr_transport_kmt_exchange(
    sagr_instance_t opaque_instance, const sagr_kmt_envelope_request_t *request,
    const sagr_kmt_call_options_t *options,
    sagr_kmt_envelope_result_t *result, int32_t *wire_status,
    sagr_error_info_t *error, uint32_t error_size) {
  struct sagr_instance *instance = (struct sagr_instance *)opaque_instance;
  sagr_kmt_call_options_t local_options;
  monotonic_deadline_t deadline;
  uint8_t request_frame[SAGR_WIRE_KMT_FRAME_BYTES];
  uint8_t response_frame[SAGR_WIRE_MAX_RECORD_BYTES];
  size_t request_size = 0;
  size_t response_size = 0;
  uint64_t request_id = 0;
  int native_errno = 0;
  int record_sent = 0;
  int32_t decoded_wire_status = -1;
  const char *reason = "KMT operation failed";
  sagr_status_t status;

  initialize_error(error, error_size);
  if (instance == NULL || instance->magic != SAGR_INSTANCE_MAGIC ||
      request == NULL || result == NULL ||
      (error != NULL && error_size < sizeof(*error))) {
    return fail_open(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "invalid KMT exchange arguments");
  }
  if (options == NULL) {
    memset(&local_options, 0, sizeof(local_options));
    local_options.struct_size = (uint32_t)sizeof(local_options);
    local_options.timeout_ns = SAGR_DEFAULT_OPEN_TIMEOUT_NS;
    local_options.cancel_fd = -1;
  } else {
    if (options->struct_size < sizeof(*options)) {
      return fail_open(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                       "KMT call options are too small");
    }
    memcpy(&local_options, options, sizeof(local_options));
  }
  if (local_options.flags != 0 || local_options.cancel_fd < -1 ||
      local_options.reserved0 != 0 ||
      !reserved_is_zero(local_options.reserved, sizeof(local_options.reserved))) {
    return fail_open(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "invalid KMT call options");
  }
  if (local_options.cancel_fd >= 0) {
    const int fd_flags = fcntl(local_options.cancel_fd, F_GETFD);
    if (fd_flags < 0 || (fd_flags & FD_CLOEXEC) == 0) {
      return fail_open(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1,
                       fd_flags < 0 ? errno : EINVAL,
                       "KMT cancellation fd must be CLOEXEC");
    }
  }
  if (make_deadline(local_options.timeout_ns,
                    local_options.absolute_deadline_ns, &deadline) != 0) {
    return fail_open(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1,
                     errno, "invalid KMT deadline");
  }
  if (!kmt_capability_selected(instance)) {
    return fail_open(error, error_size, SAGR_STATUS_NOT_SUPPORTED,
                     SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY, 0,
                     "KMT capability was not negotiated");
  }
  status = require_queue_transport(instance, error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (instance->operation_active != 0) {
    return fail_open(error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another transport operation is active");
  }
  status = sagr_protocol_allocate_request_id(&instance->next_request_id,
                                             &request_id);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(error, error_size, status, -1, 0,
                     "KMT request ID space exhausted");
  }
  status = sagr_protocol_encode_kmt_request(
      &instance->info, request_id, request, request_frame,
      sizeof(request_frame), &request_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(error, error_size, status, -1, 0,
                     "could not encode KMT request");
  }
  instance->operation_active = 1;
  status = send_record(instance->socket_fd, request_frame, request_size,
                       &deadline, local_options.cancel_fd, &native_errno,
                       &record_sent);
  if (status != SAGR_STATUS_SUCCESS) {
    if (record_sent != 0) {
      poison_queue_transport(instance);
    }
    instance->operation_active = 0;
    return fail_open(error, error_size, status, -1, native_errno,
                     "KMT request send failed");
  }
  status = receive_record(instance->socket_fd, response_frame,
                          sizeof(response_frame), &response_size, &deadline,
                          local_options.cancel_fd, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    poison_queue_transport(instance);
    instance->operation_active = 0;
    return fail_open(error, error_size, status, -1, native_errno,
                     "KMT result receive failed");
  }
  status = sagr_protocol_decode_kmt_result(
      response_frame, response_size, &instance->info, request_id, result,
      &decoded_wire_status, &reason);
  if (status != SAGR_STATUS_SUCCESS ||
      result->operation != request->operation ||
      result->operation_sequence != request->operation_sequence ||
      (request->operation != SAGR_KMT_OP_OPEN_KFD &&
       (result->owner_id != request->owner_id ||
        result->owner_generation != request->owner_generation)) ||
      ((request->operation != SAGR_KMT_OP_OPEN_KFD &&
        request->operation != SAGR_KMT_OP_TOPOLOGY_SNAPSHOT &&
        request->operation != SAGR_KMT_OP_ALLOC_MEMORY &&
        request->operation != SAGR_KMT_OP_QUEUE_CREATE &&
        request->operation != SAGR_KMT_OP_EVENT_CREATE) &&
       (result->object_id != request->object_id ||
        result->object_generation != request->object_generation)) ||
      result->auxiliary_id != request->auxiliary_id ||
      result->auxiliary_generation != request->auxiliary_generation) {
    poison_queue_transport(instance);
    instance->operation_active = 0;
    if (status == SAGR_STATUS_SUCCESS) {
      status = SAGR_STATUS_PROTOCOL_ERROR;
      reason = "KMT result identity mismatch";
    }
    return fail_open(error, error_size, status, decoded_wire_status,
                     native_errno, reason);
  }
  instance->operation_active = 0;
  if (wire_status != NULL) {
    *wire_status = decoded_wire_status;
  }
  if (error != NULL && error_size >= sizeof(*error)) {
    error->wire_status = decoded_wire_status;
    error->status = SAGR_STATUS_SUCCESS;
    (void)snprintf(error->message, sizeof(error->message), "%s", reason);
  }
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t
sagr_code_object_upload(
    sagr_instance_t opaque_instance, const void *image, size_t image_size,
    const char *kernel_name,
    const sagr_queue_operation_options_t *operation_options,
    sagr_code_object_remote_info_t *remote, uint32_t remote_size,
    sagr_error_info_t *error, uint32_t error_size)
{
  struct sagr_instance *instance = (struct sagr_instance *)opaque_instance;
  sagr_queue_operation_options_t local_options;
  monotonic_deadline_t deadline;
  sagr_code_object_info_t parsed;
  sagr_code_object_kernel_info_t kernel;
  sagr_code_object_dispatch_binding_t binding;
  sagr_wire_code_object_request_t request;
  sagr_wire_code_object_response_t response;
  uint8_t digest[32];
  uint8_t request_frame[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  size_t request_size = 0U;
  uint64_t request_id = 0U;
  uint64_t object_id = 0U;
  uint64_t generation = 0U;
  uint32_t chunk_count;
  uint32_t chunk_index;
  uint64_t image_offset;
  int native_errno = 0;
  int record_sent = 0;
  int poison = 0;
  int transaction_started = 0;
  int32_t decoded_wire_status = -1;
  sagr_status_t status = SAGR_STATUS_SUCCESS;
  const char *reason = "code-object upload failed";

  initialize_error(error, error_size);
  if (remote != NULL && remote_size >= sizeof(*remote)) {
    memset(remote, 0, sizeof(*remote));
    remote->struct_size = (uint32_t)sizeof(*remote);
  }
  if (instance == NULL || instance->magic != SAGR_INSTANCE_MAGIC ||
      image == NULL || image_size == 0U ||
      image_size > SAGR_CODE_OBJECT_TRANSPORT_MAX_IMAGE_BYTES ||
      kernel_name == NULL || kernel_name[0] == '\0' || remote == NULL ||
      (error == NULL && error_size != 0U)) {
    return fail_open(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "invalid code-object upload arguments");
  }
  if (remote_size < sizeof(*remote)) {
    if (remote_size >= sizeof(remote->struct_size)) {
      remote->struct_size = (uint32_t)sizeof(*remote);
    }
    return fail_open(error, error_size, SAGR_STATUS_BUFFER_TOO_SMALL, -1, 0,
                     "code-object remote result is too small");
  }
  memset(&parsed, 0, sizeof(parsed));
  status = sagr_code_object_validate(image, image_size, &parsed,
                                     (uint32_t)sizeof(parsed));
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(error, error_size, status, -1, 0,
                     "HSACO image failed local validation");
  }
  memset(&kernel, 0, sizeof(kernel));
  status = sagr_code_object_get_kernel(&parsed, kernel_name, &kernel,
                                       (uint32_t)sizeof(kernel));
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(error, error_size, status, -1, 0,
                     "requested HSACO kernel was not found");
  }
  memset(&binding, 0, sizeof(binding));
  status = sagr_code_object_describe_dispatch(
      &parsed, kernel_name, image_size, &binding, (uint32_t)sizeof(binding));
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(error, error_size, status, -1, 0,
                     "requested HSACO kernel could not be bound");
  }
  if (!code_object_capability_selected(instance)) {
    return fail_open(error, error_size, SAGR_STATUS_NOT_SUPPORTED,
                     SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY, 0,
                     "code-object transport capability was not negotiated");
  }
  status = require_code_object_transport(instance, error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (operation_options == NULL) {
    status = sagr_queue_operation_options_init(
        &local_options, (uint32_t)sizeof(local_options));
    if (status != SAGR_STATUS_SUCCESS) {
      return fail_open(error, error_size, status, -1, 0,
                       "could not initialize code-object operation options");
    }
  } else {
    if (operation_options->struct_size < sizeof(*operation_options)) {
      return fail_open(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                       "code-object operation options are too small");
    }
    memcpy(&local_options, operation_options, sizeof(local_options));
  }
  status = validate_queue_operation_options(&local_options);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(error, error_size, status, -1, 0,
                     "invalid code-object operation options");
  }
  if (local_options.cancel_fd >= 0) {
    const int fd_flags = fcntl(local_options.cancel_fd, F_GETFD);
    if (fd_flags < 0 || (fd_flags & FD_CLOEXEC) == 0) {
      return fail_open(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1,
                       fd_flags < 0 ? errno : EINVAL,
                       "code-object cancellation fd must be CLOEXEC");
    }
  }
  if (make_deadline(local_options.timeout_ns, local_options.absolute_deadline_ns,
                    &deadline) != 0) {
    return fail_open(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, errno,
                     "invalid code-object deadline");
  }
  if (instance->operation_active != 0) {
    return fail_open(error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another transport operation is active");
  }
  sagr_sha256(image, image_size, digest);
  chunk_count = (uint32_t)((image_size +
                            SAGR_CODE_OBJECT_TRANSPORT_CHUNK_BYTES - 1U) /
                           SAGR_CODE_OBJECT_TRANSPORT_CHUNK_BYTES);
  memset(&request, 0, sizeof(request));
  request.major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
  request.minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN;
  request.body.begin.image_size = (uint64_t)image_size;
  request.body.begin.chunk_data_bytes =
      SAGR_CODE_OBJECT_TRANSPORT_CHUNK_BYTES;
  request.body.begin.chunk_count = chunk_count;
  request.body.begin.segment_count = parsed.segment_count;
  request.body.begin.kernel_index = kernel.index;
  memcpy(request.body.begin.image_sha256, digest, sizeof(digest));
  request.body.begin.elf_machine = parsed.elf_machine;
  request.body.begin.elf_type = parsed.elf_type;
  request.body.begin.elf_osabi = parsed.elf_osabi;
  request.body.begin.elf_abi_version = parsed.elf_abi_version;
  request.body.begin.elf_flags = parsed.elf_flags;
  request.body.begin.gfx_target = parsed.gfx_target;
  request.body.begin.code_object_version = parsed.code_object_version;
  request.body.begin.metadata_major = parsed.metadata_major;
  request.body.begin.metadata_minor = parsed.metadata_minor;
  request.body.begin.relocation_count = parsed.relocation_count;
  request.body.begin.kernarg_segment_size = kernel.kernarg_segment_size;
  request.body.begin.kernarg_segment_align = kernel.kernarg_segment_align;
  request.body.begin.group_segment_fixed_size =
      kernel.group_segment_fixed_size;
  request.body.begin.private_segment_fixed_size =
      kernel.private_segment_fixed_size;
  request.body.begin.max_flat_workgroup_size = kernel.max_flat_workgroup_size;
  request.body.begin.wavefront_size = kernel.wavefront_size;
  request.body.begin.sgpr_count = kernel.sgpr_count;
  request.body.begin.vgpr_count = kernel.vgpr_count;
  request.body.begin.uses_dynamic_stack = kernel.uses_dynamic_stack;
  request.body.begin.descriptor_size = binding.descriptor_size;
  request.body.begin.descriptor_kernel_code_entry_byte_offset =
      binding.descriptor_kernel_code_entry_byte_offset;
  request.body.begin.code_address = binding.code_address;
  request.body.begin.code_file_offset = binding.code_file_offset;
  request.body.begin.code_size = binding.code_size;
  request.body.begin.descriptor_address = binding.descriptor_address;
  request.body.begin.descriptor_file_offset = binding.descriptor_file_offset;
  memcpy(request.body.begin.kernel_name, kernel.name,
         SAGR_CODE_OBJECT_NAME_BYTES);
  memcpy(request.body.begin.symbol, kernel.symbol,
         SAGR_CODE_OBJECT_NAME_BYTES);
  memcpy(request.body.begin.descriptor, kernel.descriptor,
         SAGR_CODE_OBJECT_DESCRIPTOR_BYTES);
  for (chunk_index = 0; chunk_index < parsed.segment_count; ++chunk_index) {
    request.body.begin.segments[chunk_index].type =
        parsed.segments[chunk_index].type;
    request.body.begin.segments[chunk_index].flags =
        parsed.segments[chunk_index].flags;
    request.body.begin.segments[chunk_index].file_offset =
        parsed.segments[chunk_index].file_offset;
    request.body.begin.segments[chunk_index].virtual_address =
        parsed.segments[chunk_index].virtual_address;
    request.body.begin.segments[chunk_index].file_size =
        parsed.segments[chunk_index].file_size;
    request.body.begin.segments[chunk_index].memory_size =
        parsed.segments[chunk_index].memory_size;
    request.body.begin.segments[chunk_index].alignment =
        parsed.segments[chunk_index].alignment;
  }

  instance->operation_active = 1;
  for (;;) {
    sagr_wire_code_object_response_t decoded;
    status = sagr_protocol_allocate_request_id(&instance->next_request_id,
                                               &request_id);
    if (status != SAGR_STATUS_SUCCESS) {
      reason = "code-object request ID space exhausted";
      goto upload_fail;
    }
    status = sagr_protocol_encode_code_object_request(
        &instance->info, request_id, &request, request_frame,
        sizeof(request_frame), &request_size);
    if (status != SAGR_STATUS_SUCCESS) {
      reason = "could not encode code-object BEGIN";
      goto upload_fail;
    }
    status = send_record(instance->socket_fd, request_frame, request_size,
                         &deadline, local_options.cancel_fd, &native_errno,
                         &record_sent);
    if (status != SAGR_STATUS_SUCCESS) {
      reason = "code-object BEGIN send failed";
      poison = record_sent != 0;
      goto upload_fail;
    }
    decoded_wire_status = -1;
    status = receive_code_object_ack(
        instance, &request, request_id, &deadline, local_options.cancel_fd,
        &decoded, &decoded_wire_status, &native_errno, &reason, &poison);
    if (status != SAGR_STATUS_SUCCESS) {
      goto upload_fail;
    }
    if (decoded.opcode != SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN ||
        decoded.object_id == 0U || decoded.generation == 0U ||
        decoded.accepted_offset != 0U || decoded.accepted_count != 0U ||
        decoded.chunk_index != 0U || decoded.image_size != image_size ||
        decoded.kernel_index != kernel.index ||
        decoded.segment_count != parsed.segment_count ||
        memcmp(decoded.image_sha256, digest, sizeof(digest)) != 0) {
      status = SAGR_STATUS_PROTOCOL_ERROR;
      reason = "code-object BEGIN ACK identity mismatch";
      poison = 1;
      goto upload_fail;
    }
    object_id = decoded.object_id;
    generation = decoded.generation;
    transaction_started = 1;
    break;
  }

  for (chunk_index = 0, image_offset = 0U; chunk_index < chunk_count;
       ++chunk_index) {
    const size_t remaining = image_size - (size_t)image_offset;
    const uint32_t chunk_bytes = (uint32_t)(remaining >
                                                    SAGR_CODE_OBJECT_TRANSPORT_CHUNK_BYTES
                                                ? SAGR_CODE_OBJECT_TRANSPORT_CHUNK_BYTES
                                                : remaining);
    memset(&request, 0, sizeof(request));
    request.major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
    request.minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
    request.opcode = SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK;
    request.object_id = object_id;
    request.generation = generation;
    request.image_offset = image_offset;
    request.byte_count = chunk_bytes;
    request.chunk_index = chunk_index;
    memcpy(request.body.chunk, (const uint8_t *)image + image_offset,
           chunk_bytes);
    request.chunk_crc32c = sagr_crc32c(request.body.chunk, chunk_bytes);
    status = sagr_protocol_allocate_request_id(&instance->next_request_id,
                                               &request_id);
    if (status != SAGR_STATUS_SUCCESS) {
      reason = "code-object chunk request ID space exhausted";
      goto upload_fail;
    }
    status = sagr_protocol_encode_code_object_request(
        &instance->info, request_id, &request, request_frame,
        sizeof(request_frame), &request_size);
    if (status != SAGR_STATUS_SUCCESS) {
      reason = "could not encode code-object CHUNK";
      goto upload_fail;
    }
    record_sent = 0;
    status = send_record(instance->socket_fd, request_frame, request_size,
                         &deadline, local_options.cancel_fd, &native_errno,
                         &record_sent);
    if (status != SAGR_STATUS_SUCCESS) {
      reason = "code-object CHUNK send failed";
      poison = record_sent != 0;
      goto upload_fail;
    }
    decoded_wire_status = -1;
    status = receive_code_object_ack(
        instance, &request, request_id, &deadline, local_options.cancel_fd,
        &response, &decoded_wire_status, &native_errno, &reason, &poison);
    if (status != SAGR_STATUS_SUCCESS) {
      if (decoded_wire_status >= 0 && poison == 0) {
        transaction_started = 0;
      }
      goto upload_fail;
    }
    if (response.opcode != SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK ||
        response.object_id != object_id || response.generation != generation ||
        response.accepted_offset != image_offset ||
        response.accepted_count != chunk_bytes ||
        response.chunk_index != chunk_index) {
      status = SAGR_STATUS_PROTOCOL_ERROR;
      reason = "code-object CHUNK ACK progress mismatch";
      poison = 1;
      goto upload_fail;
    }
    image_offset += chunk_bytes;
  }

  memset(&request, 0, sizeof(request));
  request.major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
  request.minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT;
  request.object_id = object_id;
  request.generation = generation;
  request.byte_count = (uint32_t)image_size;
  request.chunk_index = chunk_count;
  memcpy(request.body.commit_sha256, digest, sizeof(digest));
  if (!code_object_commit_matches_manifest(&request, image_size,
                                           chunk_count)) {
    status = SAGR_STATUS_PROTOCOL_ERROR;
    reason = "code-object COMMIT manifest mismatch";
    poison = 1;
    goto upload_fail;
  }
  status = sagr_protocol_allocate_request_id(&instance->next_request_id,
                                             &request_id);
  if (status != SAGR_STATUS_SUCCESS) {
    reason = "code-object COMMIT request ID space exhausted";
    goto upload_fail;
  }
  status = sagr_protocol_encode_code_object_request(
      &instance->info, request_id, &request, request_frame,
      sizeof(request_frame), &request_size);
  if (status != SAGR_STATUS_SUCCESS) {
    reason = "could not encode code-object COMMIT";
    goto upload_fail;
  }
  record_sent = 0;
  status = send_record(instance->socket_fd, request_frame, request_size,
                       &deadline, local_options.cancel_fd, &native_errno,
                       &record_sent);
  if (status != SAGR_STATUS_SUCCESS) {
    reason = "code-object COMMIT send failed";
    poison = record_sent != 0;
    goto upload_fail;
  }
  decoded_wire_status = -1;
  status = receive_code_object_ack(
      instance, &request, request_id, &deadline, local_options.cancel_fd,
      &response, &decoded_wire_status, &native_errno, &reason, &poison);
  if (status != SAGR_STATUS_SUCCESS) {
    if (decoded_wire_status >= 0 && poison == 0) {
      transaction_started = 0;
    }
    goto upload_fail;
  }
  if (response.opcode != SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT ||
      response.object_id != object_id || response.generation != generation ||
      response.accepted_offset != image_size ||
      response.accepted_count != (uint32_t)image_size ||
      response.chunk_index != chunk_count || response.image_size != image_size ||
      response.kernel_index != kernel.index ||
      response.segment_count != parsed.segment_count ||
      memcmp(response.image_sha256, digest, sizeof(digest)) != 0 ||
      response.mapped_base_va != 0U || response.descriptor_va != 0U ||
      response.code_va != 0U || response.kernarg_va != 0U) {
    status = SAGR_STATUS_PROTOCOL_ERROR;
    reason = "code-object COMMIT ACK identity or mapping boundary mismatch";
    poison = 1;
    goto upload_fail;
  }
  memcpy(remote->image_sha256, digest, sizeof(digest));
  remote->flags = SAGR_CODE_OBJECT_REMOTE_FLAG_STAGED_IDENTITY_ONLY;
  remote->object_id = object_id;
  remote->generation = generation;
  remote->image_size = image_size;
  remote->kernel_index = kernel.index;
  remote->segment_count = parsed.segment_count;
  transaction_started = 0;
  instance->operation_active = 0;
  if (error != NULL && error_size >= sizeof(*error)) {
    error->status = SAGR_STATUS_SUCCESS;
    error->wire_status = SAGR_WIRE_STATUS_OK;
    (void)snprintf(error->message, sizeof(error->message),
                   "%s", "code-object image staged and committed");
  }
  return SAGR_STATUS_SUCCESS;

upload_fail:
  if (transaction_started != 0) {
    poison = 1;
  }
  if (poison != 0) {
    poison_queue_transport(instance);
  }
  instance->operation_active = 0;
  return fail_open(error, error_size, status, decoded_wire_status,
                   native_errno, reason);
}

sagr_status_t sagr_instance_open(
    const char *endpoint, const sagr_instance_open_options_t *options,
    sagr_instance_t *out_instance, sagr_error_info_t *out_error,
    uint32_t error_size) {
  sagr_instance_open_options_t local_options;
  monotonic_deadline_t deadline;
  struct sockaddr_un address;
  struct ucred credentials;
  socklen_t credentials_size = (socklen_t)sizeof(credentials);
  uint8_t client_nonce[16];
  uint8_t hello[SAGR_WIRE_HELLO_FRAME_BYTES];
  uint8_t ack[SAGR_WIRE_MAX_HANDSHAKE_BYTES];
  size_t hello_size = 0;
  size_t ack_size = 0;
  size_t endpoint_size;
  uint64_t request_id = 0;
  int socket_fd = -1;
  int native_errno = 0;
  int cancel_fd_flags;
  sagr_status_t status;
  int32_t wire_status = -1;
  const char *reason = "handshake failed";
  sagr_wire_ack_result_t result;
  struct sagr_instance *instance = NULL;

  if (out_instance != NULL) {
    *out_instance = NULL;
  }
  initialize_error(out_error, error_size);
  if (out_instance == NULL || endpoint == NULL ||
      (out_error == NULL && error_size != 0)) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1,
                     0, "invalid output, endpoint, or error buffer");
  }
  if (out_error != NULL && error_size < sizeof(*out_error)) {
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  if (endpoint[0] != '/') {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1,
                     0, "endpoint must be an absolute pathname");
  }
  endpoint_size = strlen(endpoint);
  if (endpoint_size == 0 || endpoint_size >= sizeof(address.sun_path)) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1,
                     0, "endpoint pathname is empty or too long");
  }

  if (options == NULL) {
    status = sagr_instance_open_options_init(
        &local_options, (uint32_t)sizeof(local_options));
    if (status != SAGR_STATUS_SUCCESS) {
      return fail_open(out_error, error_size, status, -1, 0,
                       "could not initialize default options");
    }
  } else {
    if (options->struct_size < sizeof(*options)) {
      return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1,
                       0, "options structure is too small");
    }
    memcpy(&local_options, options, sizeof(local_options));
  }
  status = validate_options(&local_options);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid instance open options");
  }
  if (local_options.cancel_fd >= 0) {
    cancel_fd_flags = fcntl(local_options.cancel_fd, F_GETFD);
    if (cancel_fd_flags < 0) {
      return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1,
                       errno, "cancel_fd is not an open descriptor");
    }
    if ((cancel_fd_flags & FD_CLOEXEC) == 0) {
      return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1,
                       EINVAL, "cancel_fd must have FD_CLOEXEC set");
    }
  }
  if (make_deadline(local_options.open_timeout_ns,
                    local_options.absolute_deadline_ns, &deadline) != 0) {
    const int deadline_errno = errno;
    return fail_open(out_error, error_size,
                     deadline_errno == EOVERFLOW
                         ? SAGR_STATUS_INVALID_ARGUMENT
                         : SAGR_STATUS_INTERNAL_ERROR,
                     -1, deadline_errno,
                     deadline_errno == EOVERFLOW
                         ? "deadline is outside the CLOCK_MONOTONIC range"
                         : "could not read CLOCK_MONOTONIC");
  }
  status = generate_handshake_identity(&deadline, &request_id, client_nonce,
                                       local_options.cancel_fd, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "could not generate handshake identity");
  }

  socket_fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (socket_fd < 0) {
    native_errno = errno;
    status = map_connect_errno(native_errno);
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "could not create AF_UNIX SOCK_SEQPACKET socket");
  }
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  memcpy(address.sun_path, endpoint, endpoint_size + 1U);
  status = connect_once(
      socket_fd, &address,
      (socklen_t)(offsetof(struct sockaddr_un, sun_path) + endpoint_size + 1U),
      &deadline, local_options.cancel_fd, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    (void)close(socket_fd);
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "could not connect to endpoint");
  }

  status = check_operation_state(&deadline, local_options.cancel_fd,
                                 &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    (void)close(socket_fd);
    return fail_open(out_error, error_size, status, -1, native_errno,
                     status == SAGR_STATUS_CANCELLED
                         ? "handshake was cancelled"
                         : "handshake deadline expired");
  }
  if (getsockopt(socket_fd, SOL_SOCKET, SO_PEERCRED, &credentials,
                 &credentials_size) != 0 ||
      credentials_size != sizeof(credentials)) {
    native_errno = credentials_size != sizeof(credentials) ? EPROTO : errno;
    (void)close(socket_fd);
    return fail_open(out_error, error_size, SAGR_STATUS_UNAVAILABLE, -1,
                     native_errno, "could not read peer credentials");
  }
  if (credentials.uid != geteuid()) {
    (void)close(socket_fd);
    return fail_open(out_error, error_size, SAGR_STATUS_UNAUTHORIZED, -1, 0,
                     "endpoint peer UID differs from runtime effective UID");
  }

  status = sagr_protocol_encode_hello(&local_options, request_id, client_nonce,
                                      hello, sizeof(hello), &hello_size);
  if (status == SAGR_STATUS_SUCCESS) {
    status = send_record(socket_fd, hello, hello_size, &deadline,
                         local_options.cancel_fd, &native_errno, NULL);
  }
  if (status == SAGR_STATUS_SUCCESS) {
    status = receive_record(socket_fd, ack, sizeof(ack), &ack_size, &deadline,
                            local_options.cancel_fd, &native_errno);
  }
  if (status != SAGR_STATUS_SUCCESS) {
    (void)close(socket_fd);
    return fail_open(out_error, error_size, status, -1, native_errno,
                     status == SAGR_STATUS_TIMED_OUT
                         ? "handshake deadline expired"
                         : (status == SAGR_STATUS_CANCELLED
                                ? "handshake was cancelled"
                                : "handshake record I/O failed"));
  }

  status = check_operation_state(&deadline, local_options.cancel_fd,
                                 &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    reason = status == SAGR_STATUS_CANCELLED
                 ? "handshake was cancelled before ACK validation"
                 : "handshake deadline expired before ACK validation";
  }
  if (status == SAGR_STATUS_SUCCESS) {
    status = sagr_protocol_decode_ack(ack, ack_size, &local_options, request_id,
                                      client_nonce, &result, &wire_status,
                                      &reason);
  }
  if (status == SAGR_STATUS_SUCCESS) {
    status = check_operation_state(&deadline, local_options.cancel_fd,
                                   &native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      wire_status = -1;
      reason = status == SAGR_STATUS_CANCELLED
                   ? "handshake was cancelled during ACK validation"
                   : "handshake deadline expired during ACK validation";
    }
  }
  if (status != SAGR_STATUS_SUCCESS) {
    (void)close(socket_fd);
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }

  instance = (struct sagr_instance *)calloc(1, sizeof(*instance));
  if (instance == NULL) {
    native_errno = errno;
    (void)close(socket_fd);
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1,
                     native_errno, "could not allocate runtime instance");
  }
  instance->magic = SAGR_INSTANCE_MAGIC;
  instance->socket_fd = socket_fd;
  instance->next_request_id =
      request_id == UINT64_MAX ? 0 : request_id + UINT64_C(1);
  instance->info.struct_size = (uint32_t)sizeof(instance->info);
  instance->info.selected_version_major = result.selected_major;
  instance->info.selected_version_minor = result.selected_minor;
  instance->info.maximum_record_bytes = result.maximum_record_bytes;
  memcpy(instance->info.negotiated_capabilities,
         result.selected_capabilities,
         sizeof(instance->info.negotiated_capabilities));
  memcpy(instance->info.daemon_uuid, result.daemon_uuid, 16);
  memcpy(instance->info.job_uuid, result.job_uuid, 16);
  instance->info.connection_id = result.connection_id;
  instance->info.epoch = result.epoch;
  instance->info.rank = result.rank;
  instance->info.world_size = result.world_size;
  instance->info.peer_uid = (uint32_t)credentials.uid;
  instance->info.peer_pid = credentials.pid < 0 ? 0U : (uint32_t)credentials.pid;
  instance->info.request_id = result.request_id;
  *out_instance = instance;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_instance_get_info(sagr_instance_t instance,
                                     sagr_instance_info_t *info,
                                     uint32_t info_size) {
  if (instance == NULL || instance->magic != SAGR_INSTANCE_MAGIC) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memcpy(info, &instance->info, sizeof(*info));
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_instance_close(sagr_instance_t *instance) {
  struct sagr_instance *owned_instance;
  struct sagr_queue *queue;
  struct sagr_memory *memory;
  struct sagr_signal *signal;
  if (instance == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (*instance == NULL) {
    return SAGR_STATUS_SUCCESS;
  }
  owned_instance = *instance;
  if (owned_instance->magic != SAGR_INSTANCE_MAGIC) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  *instance = NULL;
  owned_instance->magic = 0;
  if (owned_instance->socket_fd >= 0) {
    (void)close(owned_instance->socket_fd);
    owned_instance->socket_fd = -1;
  }
  if (owned_instance->pending_dispatch != NULL) {
    memset(owned_instance->pending_dispatch, 0,
           sizeof(*owned_instance->pending_dispatch));
    free(owned_instance->pending_dispatch);
    owned_instance->pending_dispatch = NULL;
  }
  queue = owned_instance->queues;
  while (queue != NULL) {
    struct sagr_queue *next = queue->next;
    queue->magic = 0;
    queue->instance = NULL;
    memset(queue, 0, sizeof(*queue));
    free(queue);
    queue = next;
  }
  owned_instance->queues = NULL;
  owned_instance->queue_count = 0;
  memory = owned_instance->memories;
  while (memory != NULL) {
    struct sagr_memory *next = memory->next;
    memory->magic = 0;
    memory->instance = NULL;
    memset(memory, 0, sizeof(*memory));
    free(memory);
    memory = next;
  }
  owned_instance->memories = NULL;
  owned_instance->memory_count = 0;
  signal = owned_instance->signals;
  while (signal != NULL) {
    struct sagr_signal *next = signal->next;
    signal->magic = 0;
    signal->instance = NULL;
    memset(signal, 0, sizeof(*signal));
    free(signal);
    signal = next;
  }
  owned_instance->signals = NULL;
  owned_instance->signal_count = 0;
  owned_instance->pending_signal_wait_count = 0;
  memset(&owned_instance->info, 0, sizeof(owned_instance->info));
  free(owned_instance);
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_error_output(sagr_error_info_t *error,
                                           uint32_t error_size) {
  initialize_error(error, error_size);
  if (error == NULL && error_size != 0) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (error != NULL && error_size < sizeof(*error)) {
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t prepare_sized_output(void *output, uint32_t output_size,
                                          size_t required_size,
                                          int required) {
  size_t clear_size;
  if (output == NULL) {
    return required != 0 || output_size != 0 ? SAGR_STATUS_INVALID_ARGUMENT
                                             : SAGR_STATUS_SUCCESS;
  }
  clear_size = output_size < required_size ? output_size : required_size;
  memset(output, 0, clear_size);
  if (output_size >= sizeof(uint32_t)) {
    const uint32_t encoded_size = (uint32_t)required_size;
    memcpy(output, &encoded_size, sizeof(encoded_size));
  }
  return (size_t)output_size < required_size ? SAGR_STATUS_BUFFER_TOO_SMALL
                                              : SAGR_STATUS_SUCCESS;
}

static void fill_queue_info(const struct sagr_queue *queue,
                            sagr_queue_info_t *info) {
  info->struct_size = (uint32_t)sizeof(*info);
  info->depth = queue->depth;
  info->queue_id = queue->queue_id;
  info->generation = queue->generation;
  info->connection_id = queue->instance->info.connection_id;
  info->epoch = queue->instance->info.epoch;
  memcpy(info->daemon_uuid, queue->instance->info.daemon_uuid,
         sizeof(info->daemon_uuid));
}

sagr_status_t sagr_queue_create(
    sagr_instance_t instance, const sagr_queue_create_options_t *options,
    const sagr_queue_operation_options_t *operation_options,
    sagr_queue_t *out_queue, sagr_queue_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_queue_create_options_t local_create;
  sagr_queue_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_queue_request_t request;
  sagr_wire_queue_response_t response;
  struct sagr_queue *queue;
  sagr_status_t status;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "queue creation failed";

  if (out_queue != NULL) {
    *out_queue = NULL;
  }
  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_sized_output(out_info, info_size, sizeof(*out_info), 0);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid queue info output buffer");
  }
  if (instance == NULL || instance->magic != SAGR_INSTANCE_MAGIC ||
      out_queue == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid instance or queue output");
  }
  if (options == NULL) {
    status = sagr_queue_create_options_init(
        &local_create, (uint32_t)sizeof(local_create));
  } else if (options->struct_size < sizeof(*options)) {
    status = SAGR_STATUS_INVALID_ARGUMENT;
  } else {
    memcpy(&local_create, options, sizeof(local_create));
    status = validate_queue_create_options(&local_create);
  }
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid queue create options");
  }
  status = prepare_queue_operation(operation_options, &local_operation,
                                   &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid queue operation options");
  }
  if (!queue_capability_selected(instance)) {
    return fail_open(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED, -1, 0,
                     "QUEUE_CONTROL_V1 was not negotiated");
  }
  status = require_queue_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (instance->queue_count >= 8U) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1, 0,
                     "runtime queue limit reached");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another queue operation is active");
  }
  queue = (struct sagr_queue *)calloc(1, sizeof(*queue));
  if (queue == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1,
                     errno, "could not allocate queue handle");
  }

  memset(&request, 0, sizeof(request));
  request.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  request.minor = SAGR_QUEUE_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_QUEUE_OPCODE_CREATE;
  request.arg0 = local_create.depth;
  instance->operation_active = 1;
  status = exchange_queue_request(instance, &request, &deadline,
                                  local_operation.cancel_fd, &response,
                                  &wire_status, &native_errno, &reason, NULL);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    free(queue);
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.queue_id == 0 || response.generation == 0 ||
      response.sequence != 0 || response.value != local_create.depth ||
      response.error_code != 0 ||
      queue_id_is_active(instance, response.queue_id)) {
    poison_queue_transport(instance);
    free(queue);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "invalid successful CREATE ACK");
  }

  queue->magic = SAGR_QUEUE_MAGIC;
  queue->instance = instance;
  queue->queue_id = response.queue_id;
  queue->generation = response.generation;
  queue->next_sequence = 1;
  queue->depth = local_create.depth;
  queue->next = instance->queues;
  instance->queues = queue;
  ++instance->queue_count;
  if (out_info != NULL) {
    fill_queue_info(queue, out_info);
  }
  *out_queue = queue;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_queue_ring_doorbell(
    sagr_queue_t queue, uint64_t command_kind,
    const sagr_queue_operation_options_t *operation_options,
    uint64_t *out_sequence, sagr_error_info_t *out_error,
    uint32_t error_size) {
  sagr_queue_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_queue_request_t request;
  sagr_wire_queue_response_t response;
  struct sagr_instance *instance;
  sagr_status_t status;
  uint64_t sequence;
  uint64_t request_id = 0;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "doorbell notification failed";

  if (out_sequence != NULL) {
    *out_sequence = 0;
  }
  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (queue == NULL || queue->magic != SAGR_QUEUE_MAGIC ||
      queue->instance == NULL ||
      queue->instance->magic != SAGR_INSTANCE_MAGIC || out_sequence == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid queue handle or sequence output");
  }
  if (command_kind != SAGR_QUEUE_COMMAND_NOOP &&
      command_kind != SAGR_QUEUE_COMMAND_CONTROL_TEST &&
      command_kind != SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "unsupported control-only command kind");
  }
  status = prepare_queue_operation(operation_options, &local_operation,
                                   &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid queue operation options");
  }
  instance = queue->instance;
  if (!queue_capability_selected(instance)) {
    return fail_open(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED, -1, 0,
                     "QUEUE_CONTROL_V1 was not negotiated");
  }
  status = require_queue_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (instance->pending_dispatch != NULL &&
      instance->pending_dispatch->queue == queue) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "queue participates in a pinned dispatch");
  }
  if (queue->pending_count >= SAGR_QUEUE_MAX_INFLIGHT) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1, 0,
                     "runtime queue in-flight limit reached");
  }
  if (queue->next_sequence == 0 || instance->operation_active != 0) {
    return fail_open(out_error, error_size,
                     queue->next_sequence == 0 ? SAGR_STATUS_OUT_OF_RESOURCES
                                               : SAGR_STATUS_BUSY,
                     -1, 0, queue->next_sequence == 0
                                    ? "queue sequence space exhausted"
                                    : "another queue operation is active");
  }
  sequence = queue->next_sequence;
  memset(&request, 0, sizeof(request));
  request.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  request.minor = SAGR_QUEUE_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_QUEUE_OPCODE_DOORBELL;
  request.queue_id = queue->queue_id;
  request.generation = queue->generation;
  request.sequence = sequence;
  request.arg0 = command_kind;
  instance->operation_active = 1;
  status = exchange_queue_request(instance, &request, &deadline,
                                  local_operation.cancel_fd, &response,
                                  &wire_status, &native_errno, &reason,
                                  &request_id);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.queue_id != queue->queue_id ||
      response.generation != queue->generation ||
      response.sequence != sequence || response.value != 0 ||
      response.error_code != 0 || response.sim_tick == UINT64_MAX) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "invalid successful DOORBELL ACK");
  }
  queue->pending_sequences[queue->pending_count] = sequence;
  queue->pending_kinds[queue->pending_count] = command_kind;
  queue->pending_request_ids[queue->pending_count] = request_id;
  queue->pending_ack_ticks[queue->pending_count] = response.sim_tick;
  ++queue->pending_count;
  queue->next_sequence = sequence == UINT64_MAX ? 0 : sequence + 1;
  *out_sequence = sequence;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_queue_wait(
    sagr_queue_t queue, uint64_t sequence,
    const sagr_queue_operation_options_t *operation_options,
    sagr_queue_completion_t *out_completion, uint32_t completion_size,
    sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_queue_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_queue_response_t completion;
  struct sagr_instance *instance;
  sagr_status_t status;
  uint64_t command_kind = 0;
  uint64_t request_id = 0;
  uint64_t ack_tick = 0;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "queue completion failed";

  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_sized_output(out_completion, completion_size,
                                sizeof(*out_completion), 1);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid queue completion output buffer");
  }
  if (queue == NULL || queue->magic != SAGR_QUEUE_MAGIC ||
      queue->instance == NULL ||
      queue->instance->magic != SAGR_INSTANCE_MAGIC) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid queue handle");
  }
  instance = queue->instance;
  status = require_queue_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (sequence == 0 ||
      !queue_pending_kind(queue, sequence, &command_kind, &request_id,
                          &ack_tick)) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "sequence is not pending on this queue");
  }
  status = prepare_queue_operation(operation_options, &local_operation,
                                   &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid queue operation options");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another queue operation is active");
  }
  instance->operation_active = 1;
  memset(&completion, 0, sizeof(completion));
  status = receive_queue_completion(
      queue, sequence, request_id, command_kind, ack_tick, &deadline,
      local_operation.cancel_fd, &completion, &wire_status, &native_errno,
      &reason);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS && wire_status < 0) {
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (completion.opcode != SAGR_WIRE_QUEUE_OPCODE_DOORBELL ||
      completion.queue_id != queue->queue_id ||
      completion.generation != queue->generation ||
      completion.sequence != sequence) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "queue completion identity mismatch");
  }
  if (!queue_completion_is_canonical(&completion, command_kind, ack_tick)) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "queue completion result mismatch");
  }
  out_completion->status = status;
  out_completion->wire_status = wire_status;
  out_completion->queue_id = completion.queue_id;
  out_completion->generation = completion.generation;
  out_completion->sequence = completion.sequence;
  out_completion->value = completion.value;
  out_completion->error_code = completion.error_code;
  out_completion->sim_tick = completion.sim_tick;
  queue_remove_pending(queue, sequence);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_queue_destroy(
    sagr_queue_t *queue,
    const sagr_queue_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_queue_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_queue_request_t request;
  sagr_wire_queue_response_t response;
  struct sagr_instance *instance;
  struct sagr_queue **link;
  struct sagr_queue *owned_queue;
  sagr_status_t status;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "queue destruction failed";

  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (queue == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "queue pointer is null");
  }
  if (*queue == NULL) {
    return SAGR_STATUS_SUCCESS;
  }
  owned_queue = *queue;
  if (owned_queue->magic != SAGR_QUEUE_MAGIC ||
      owned_queue->instance == NULL ||
      owned_queue->instance->magic != SAGR_INSTANCE_MAGIC) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid queue handle");
  }
  instance = owned_queue->instance;
  status = require_queue_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_queue_operation(operation_options, &local_operation,
                                   &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid queue operation options");
  }
  if (owned_queue->pending_count != 0 || owned_queue->buffered_count != 0 ||
      (instance->pending_dispatch != NULL &&
       instance->pending_dispatch->queue == owned_queue)) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "queue has pending completions");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another queue operation is active");
  }
  memset(&request, 0, sizeof(request));
  request.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  request.minor = SAGR_QUEUE_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_QUEUE_OPCODE_DESTROY;
  request.queue_id = owned_queue->queue_id;
  request.generation = owned_queue->generation;
  instance->operation_active = 1;
  status = exchange_queue_request(instance, &request, &deadline,
                                  local_operation.cancel_fd, &response,
                                  &wire_status, &native_errno, &reason, NULL);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.queue_id != owned_queue->queue_id ||
      response.generation != owned_queue->generation ||
      response.sequence != 0 || response.value != 0 ||
      response.error_code != 0) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "invalid successful DESTROY ACK");
  }
  link = &instance->queues;
  while (*link != NULL && *link != owned_queue) {
    link = &(*link)->next;
  }
  if (*link != owned_queue || instance->queue_count == 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_INTERNAL_ERROR,
                     wire_status, 0, "queue ownership list is inconsistent");
  }
  *link = owned_queue->next;
  --instance->queue_count;
  *queue = NULL;
  memset(owned_queue, 0, sizeof(*owned_queue));
  free(owned_queue);
  return SAGR_STATUS_SUCCESS;
}

static void fill_memory_info(const struct sagr_memory *memory,
                             sagr_memory_info_t *info) {
  info->struct_size = (uint32_t)sizeof(*info);
  info->allocation_id = memory->allocation_id;
  info->generation = memory->generation;
  info->simulated_va = memory->simulated_va;
  info->size_bytes = memory->size_bytes;
  info->alignment_bytes = memory->alignment_bytes;
  info->connection_id = memory->instance->info.connection_id;
  info->epoch = memory->instance->info.epoch;
  memcpy(info->daemon_uuid, memory->instance->info.daemon_uuid,
         sizeof(info->daemon_uuid));
}

static sagr_status_t map_staging_errno(int error_number) {
  switch (error_number) {
    case EMFILE:
    case ENFILE:
    case ENOMEM:
    case ENOSPC:
      return SAGR_STATUS_OUT_OF_RESOURCES;
    case ENOSYS:
      return SAGR_STATUS_NOT_SUPPORTED;
    default:
      return SAGR_STATUS_INTERNAL_ERROR;
  }
}

static int write_all_at(int fd, const uint8_t *bytes, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    const size_t remaining = size - offset;
    const size_t chunk = remaining > (size_t)SSIZE_MAX
                             ? (size_t)SSIZE_MAX
                             : remaining;
    const ssize_t count = pwrite(fd, bytes + offset, chunk, (off_t)offset);
    if (count > 0) {
      offset += (size_t)count;
      continue;
    }
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count == 0) {
      errno = EIO;
    }
    return -1;
  }
  return 0;
}

static int read_all_at(int fd, uint8_t *bytes, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    const size_t remaining = size - offset;
    const size_t chunk = remaining > (size_t)SSIZE_MAX
                             ? (size_t)SSIZE_MAX
                             : remaining;
    const ssize_t count = pread(fd, bytes + offset, chunk, (off_t)offset);
    if (count > 0) {
      offset += (size_t)count;
      continue;
    }
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count == 0) {
      errno = EIO;
    }
    return -1;
  }
  return 0;
}

static int validate_staging_fd(int fd, uint64_t byte_count,
                               int expected_access, int expected_seals) {
  struct stat attributes;
  const int descriptor_flags = fcntl(fd, F_GETFD);
  const int status_flags = fcntl(fd, F_GETFL);
  const int seals = fcntl(fd, F_GET_SEALS);
  if (descriptor_flags < 0 || status_flags < 0 || seals < 0 ||
      fstat(fd, &attributes) != 0) {
    return -1;
  }
  if (!S_ISREG(attributes.st_mode) || attributes.st_nlink != 0 ||
      attributes.st_uid != geteuid() ||
      (attributes.st_mode & (mode_t)07777) != (mode_t)0600 ||
      attributes.st_size < 0 ||
      (uint64_t)attributes.st_size != byte_count ||
      (descriptor_flags & FD_CLOEXEC) == 0 ||
      (status_flags & O_ACCMODE) != expected_access || seals != expected_seals) {
    errno = EPERM;
    return -1;
  }
  return 0;
}

static sagr_status_t create_staging_fd(uint64_t byte_count, int d2h,
                                       const void *source, int *out_fd,
                                       uint32_t *out_crc,
                                       int *native_errno) {
  const int initial_seals = F_SEAL_SHRINK | F_SEAL_GROW;
  const int final_seals =
      initial_seals | F_SEAL_WRITE | F_SEAL_SEAL;
  int fd = -1;
  int seals;
  *out_fd = -1;
  *out_crc = 0;
  if (byte_count == 0 || byte_count > SAGR_MEMORY_MAX_TRANSFER_BYTES ||
      byte_count > (uint64_t)SIZE_MAX || byte_count > (uint64_t)INT64_MAX ||
      (d2h == 0 && source == NULL)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  fd = memfd_create(d2h != 0 ? "sagr-d2h" : "sagr-h2d",
                    MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (fd < 0 || fchmod(fd, (mode_t)0600) != 0 ||
      ftruncate(fd, (off_t)byte_count) != 0) {
    *native_errno = errno;
    if (fd >= 0) {
      (void)close(fd);
    }
    return map_staging_errno(*native_errno);
  }
  if (d2h == 0) {
    if (write_all_at(fd, (const uint8_t *)source, (size_t)byte_count) != 0) {
      *native_errno = errno;
      (void)close(fd);
      return map_staging_errno(*native_errno);
    }
    *out_crc = sagr_crc32c((const uint8_t *)source, (size_t)byte_count);
    seals = final_seals;
  } else {
    seals = initial_seals;
  }
  if (fcntl(fd, F_ADD_SEALS, seals) != 0 ||
      validate_staging_fd(fd, byte_count, O_RDWR, seals) != 0) {
    *native_errno = errno;
    (void)close(fd);
    return map_staging_errno(*native_errno);
  }
  *out_fd = fd;
  return SAGR_STATUS_SUCCESS;
}

static int memory_handle_is_valid(const struct sagr_memory *memory) {
  return memory != NULL && memory->magic == SAGR_MEMORY_MAGIC &&
         memory->instance != NULL &&
         memory->instance->magic == SAGR_INSTANCE_MAGIC;
}

sagr_status_t sagr_memory_allocate(
    sagr_instance_t instance, const sagr_memory_allocate_options_t *options,
    const sagr_memory_operation_options_t *operation_options,
    sagr_memory_t *out_memory, sagr_memory_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_memory_allocate_options_t local_allocate;
  sagr_memory_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_memory_request_t request;
  sagr_wire_memory_response_t response;
  struct sagr_memory *memory;
  sagr_status_t status;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "memory allocation failed";

  if (out_memory != NULL) {
    *out_memory = NULL;
  }
  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_sized_output(out_info, info_size, sizeof(*out_info), 0);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid memory info output buffer");
  }
  if (instance == NULL || instance->magic != SAGR_INSTANCE_MAGIC ||
      out_memory == NULL || options == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "invalid instance, allocation options, or output");
  }
  if (options->struct_size < sizeof(*options)) {
    status = SAGR_STATUS_INVALID_ARGUMENT;
  } else {
    memcpy(&local_allocate, options, sizeof(local_allocate));
    status = validate_memory_allocate_options(&local_allocate);
  }
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid memory allocation options");
  }
  status = prepare_memory_operation(operation_options, &local_operation,
                                    &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid memory operation options");
  }
  if (!memory_capability_selected(instance)) {
    return fail_open(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED, -1, 0,
                     "SIMULATED_MEMORY_V1 was not negotiated");
  }
  status = require_memory_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (instance->memory_count >= SAGR_MEMORY_MAX_LIVE_ALLOCATIONS) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1, 0,
                     "runtime allocation-handle limit reached");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another instance operation is active");
  }
  memory = (struct sagr_memory *)calloc(1, sizeof(*memory));
  if (memory == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1,
                     errno, "could not allocate memory handle");
  }
  memset(&request, 0, sizeof(request));
  request.major = SAGR_MEMORY_PROTOCOL_MAJOR;
  request.minor = SAGR_MEMORY_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_MEMORY_OPCODE_ALLOC;
  request.byte_count = local_allocate.size_bytes;
  request.argument = local_allocate.alignment_bytes;
  instance->operation_active = 1;
  status = exchange_memory_request(
      instance, &request, -1, &deadline, local_operation.cancel_fd, &response,
      &wire_status, &native_errno, &reason);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    free(memory);
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.allocation_id == 0 || response.generation == 0 ||
      response.allocation_id > SAGR_MEMORY_MAX_LIVE_ALLOCATIONS ||
      response.value0 !=
          SAGR_MEMORY_SIMULATED_VA_BASE +
              (response.allocation_id - UINT64_C(1)) *
                  SAGR_MEMORY_SIMULATED_VA_STRIDE ||
      response.value1 != local_allocate.size_bytes ||
      response.value2 != local_allocate.alignment_bytes ||
      response.value0 % local_allocate.alignment_bytes != 0 ||
      response.value0 > UINT64_MAX - local_allocate.size_bytes ||
      memory_id_is_active(instance, response.allocation_id) ||
      memory_range_overlaps(instance, response.value0,
                            local_allocate.size_bytes)) {
    poison_queue_transport(instance);
    free(memory);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "invalid successful ALLOC ACK");
  }
  memory->magic = SAGR_MEMORY_MAGIC;
  memory->instance = instance;
  memory->allocation_id = response.allocation_id;
  memory->generation = response.generation;
  memory->simulated_va = response.value0;
  memory->size_bytes = response.value1;
  memory->alignment_bytes = response.value2;
  memory->next = instance->memories;
  instance->memories = memory;
  ++instance->memory_count;
  if (out_info != NULL) {
    fill_memory_info(memory, out_info);
  }
  *out_memory = memory;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_memory_get_info(sagr_memory_t memory,
                                   sagr_memory_info_t *info,
                                   uint32_t info_size) {
  if (info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    memset(info, 0, info_size);
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(info, 0, sizeof(*info));
  if (!memory_handle_is_valid(memory)) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  fill_memory_info(memory, info);
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t memory_copy(
    sagr_memory_t memory, uint64_t offset, const void *source,
    void *destination, uint64_t byte_count, int d2h,
    const sagr_memory_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_memory_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_memory_request_t request;
  sagr_wire_memory_response_t response;
  struct sagr_instance *instance;
  sagr_status_t status;
  int staging_fd = -1;
  uint32_t content_crc = 0;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = d2h != 0 ? "D2H copy failed" : "H2D copy failed";

  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (!memory_handle_is_valid(memory)) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid memory handle");
  }
  if (byte_count == 0 || (d2h != 0 ? destination == NULL : source == NULL) ||
      offset > memory->size_bytes ||
      byte_count > memory->size_bytes - offset) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "invalid memory copy range or host buffer");
  }
  if (byte_count > SAGR_MEMORY_MAX_TRANSFER_BYTES) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1,
                     0, "memory transfer exceeds the runtime ceiling");
  }
  status = prepare_memory_operation(operation_options, &local_operation,
                                    &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid memory operation options");
  }
  instance = memory->instance;
  if (!memory_capability_selected(instance)) {
    return fail_open(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED, -1, 0,
                     "SIMULATED_MEMORY_V1 was not negotiated");
  }
  status = require_memory_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (instance->pending_dispatch != NULL &&
      (instance->pending_dispatch->input == memory ||
       instance->pending_dispatch->output == memory)) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "allocation participates in a pinned dispatch");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another instance operation is active");
  }
  status = create_staging_fd(byte_count, d2h, source, &staging_fd,
                             &content_crc, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "could not prepare sealed memory staging");
  }
  status = check_operation_state(&deadline, local_operation.cancel_fd,
                                 &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    (void)close(staging_fd);
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "memory operation cancelled or expired before send");
  }
  memset(&request, 0, sizeof(request));
  request.major = SAGR_MEMORY_PROTOCOL_MAJOR;
  request.minor = SAGR_MEMORY_PROTOCOL_MINOR;
  request.opcode = d2h != 0 ? SAGR_WIRE_MEMORY_OPCODE_COPY_D2H
                            : SAGR_WIRE_MEMORY_OPCODE_COPY_H2D;
  request.allocation_id = memory->allocation_id;
  request.generation = memory->generation;
  request.offset = offset;
  request.byte_count = byte_count;
  request.argument = d2h != 0 ? 0 : content_crc;
  instance->operation_active = 1;
  status = exchange_memory_request(
      instance, &request, staging_fd, &deadline, local_operation.cancel_fd,
      &response, &wire_status, &native_errno, &reason);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    (void)close(staging_fd);
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.allocation_id != memory->allocation_id ||
      response.generation != memory->generation || response.value0 != offset ||
      response.value1 != byte_count || response.value2 > UINT32_MAX ||
      (d2h == 0 && response.value2 != content_crc)) {
    poison_queue_transport(instance);
    (void)close(staging_fd);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "invalid successful memory copy ACK");
  }
  if (d2h != 0) {
    const int final_seals = F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE |
                            F_SEAL_SEAL;
    uint8_t *scratch;
    uint32_t returned_crc;
    if (validate_staging_fd(staging_fd, byte_count, O_RDWR, final_seals) != 0) {
      native_errno = errno;
      poison_queue_transport(instance);
      (void)close(staging_fd);
      return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                       wire_status, native_errno,
                       "D2H staging was not finalized canonically");
    }
    scratch = (uint8_t *)malloc((size_t)byte_count);
    if (scratch == NULL) {
      native_errno = errno;
      (void)close(staging_fd);
      return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES,
                       wire_status, native_errno,
                       "could not allocate private D2H scratch");
    }
    if (read_all_at(staging_fd, scratch, (size_t)byte_count) != 0) {
      native_errno = errno;
      free(scratch);
      (void)close(staging_fd);
      return fail_open(out_error, error_size, SAGR_STATUS_INTERNAL_ERROR,
                       wire_status, native_errno,
                       "could not read finalized D2H staging");
    }
    returned_crc = sagr_crc32c(scratch, (size_t)byte_count);
    if (returned_crc != (uint32_t)response.value2) {
      free(scratch);
      poison_queue_transport(instance);
      (void)close(staging_fd);
      return fail_open(out_error, error_size, SAGR_STATUS_CHECKSUM_ERROR,
                       wire_status, 0, "D2H staging content CRC mismatch");
    }
    status = check_operation_state(&deadline, local_operation.cancel_fd,
                                   &native_errno);
    if (status != SAGR_STATUS_SUCCESS) {
      free(scratch);
      (void)close(staging_fd);
      return fail_open(out_error, error_size, status, wire_status,
                       native_errno,
                       "D2H operation cancelled or expired after ACK");
    }
    memcpy(destination, scratch, (size_t)byte_count);
    free(scratch);
  }
  (void)close(staging_fd);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_memory_copy_from_host(
    sagr_memory_t memory, uint64_t offset, const void *source,
    uint64_t byte_count,
    const sagr_memory_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size) {
  return memory_copy(memory, offset, source, NULL, byte_count, 0,
                     operation_options, out_error, error_size);
}

sagr_status_t sagr_memory_copy_to_host(
    sagr_memory_t memory, uint64_t offset, void *destination,
    uint64_t byte_count,
    const sagr_memory_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size) {
  return memory_copy(memory, offset, NULL, destination, byte_count, 1,
                     operation_options, out_error, error_size);
}

sagr_status_t sagr_memory_free(
    sagr_memory_t *memory,
    const sagr_memory_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_memory_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_memory_request_t request;
  sagr_wire_memory_response_t response;
  struct sagr_memory *owned_memory;
  struct sagr_memory **link;
  struct sagr_instance *instance;
  sagr_status_t status;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "memory free failed";

  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (memory == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "memory pointer is null");
  }
  if (*memory == NULL) {
    return SAGR_STATUS_SUCCESS;
  }
  owned_memory = *memory;
  if (!memory_handle_is_valid(owned_memory)) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid memory handle");
  }
  instance = owned_memory->instance;
  status = require_memory_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_memory_operation(operation_options, &local_operation,
                                    &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid memory operation options");
  }
  if (instance->pending_dispatch != NULL &&
      (instance->pending_dispatch->input == owned_memory ||
       instance->pending_dispatch->output == owned_memory)) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "allocation participates in a pinned dispatch");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another instance operation is active");
  }
  memset(&request, 0, sizeof(request));
  request.major = SAGR_MEMORY_PROTOCOL_MAJOR;
  request.minor = SAGR_MEMORY_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_MEMORY_OPCODE_FREE;
  request.allocation_id = owned_memory->allocation_id;
  request.generation = owned_memory->generation;
  instance->operation_active = 1;
  status = exchange_memory_request(
      instance, &request, -1, &deadline, local_operation.cancel_fd, &response,
      &wire_status, &native_errno, &reason);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.allocation_id != owned_memory->allocation_id ||
      response.generation != owned_memory->generation || response.value0 != 0 ||
      response.value1 != 0 || response.value2 != 0) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "invalid successful FREE ACK");
  }
  link = &instance->memories;
  while (*link != NULL && *link != owned_memory) {
    link = &(*link)->next;
  }
  if (*link != owned_memory || instance->memory_count == 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_INTERNAL_ERROR,
                     wire_status, 0, "memory ownership list is inconsistent");
  }
  *link = owned_memory->next;
  --instance->memory_count;
  *memory = NULL;
  memset(owned_memory, 0, sizeof(*owned_memory));
  free(owned_memory);
  return SAGR_STATUS_SUCCESS;
}

static int signal_handle_is_valid(const struct sagr_signal *signal) {
  return signal != NULL && signal->magic == SAGR_SIGNAL_MAGIC &&
         signal->instance != NULL &&
         signal->instance->magic == SAGR_INSTANCE_MAGIC;
}

static void fill_signal_info(const struct sagr_signal *signal,
                             sagr_signal_info_t *info) {
  info->struct_size = (uint32_t)sizeof(*info);
  info->signal_id = signal->signal_id;
  info->generation = signal->generation;
  info->value = signal->value;
  info->connection_id = signal->instance->info.connection_id;
  info->epoch = signal->instance->info.epoch;
  memcpy(info->daemon_uuid, signal->instance->info.daemon_uuid,
         sizeof(info->daemon_uuid));
}

static void initialize_signal_request_for_handle(
    sagr_wire_signal_request_t *request, uint16_t opcode,
    const struct sagr_signal *signal) {
  memset(request, 0, sizeof(*request));
  request->major = SAGR_SIGNAL_PROTOCOL_MAJOR;
  request->minor = SAGR_SIGNAL_PROTOCOL_MINOR;
  request->opcode = opcode;
  if (signal != NULL) {
    request->signal_id = signal->signal_id;
    request->generation = signal->generation;
  }
}

sagr_status_t sagr_signal_create(
    sagr_instance_t instance, const sagr_signal_create_options_t *options,
    const sagr_signal_operation_options_t *operation_options,
    sagr_signal_t *out_signal, sagr_signal_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_signal_create_options_t local_create;
  sagr_signal_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_signal_request_t request;
  sagr_wire_signal_response_t response;
  struct sagr_signal *signal;
  sagr_status_t status;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "signal creation failed";

  if (out_signal != NULL) {
    *out_signal = NULL;
  }
  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_sized_output(out_info, info_size, sizeof(*out_info), 0);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid signal info output buffer");
  }
  if (instance == NULL || instance->magic != SAGR_INSTANCE_MAGIC ||
      out_signal == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid instance or signal output");
  }
  if (options == NULL) {
    status = sagr_signal_create_options_init(
        &local_create, (uint32_t)sizeof(local_create));
  } else if (options->struct_size < sizeof(*options)) {
    status = SAGR_STATUS_INVALID_ARGUMENT;
  } else {
    memcpy(&local_create, options, sizeof(local_create));
    status = validate_signal_create_options(&local_create);
  }
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid signal create options");
  }
  status = prepare_signal_operation(operation_options, &local_operation,
                                    &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid signal operation options");
  }
  if (!signal_capability_selected(instance)) {
    return fail_open(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED, -1, 0,
                     "SIGNAL_EVENT_V1 was not negotiated");
  }
  status = require_signal_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (instance->signal_count >= SAGR_SIGNAL_MAX_LIVE_SIGNALS) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1, 0,
                     "runtime signal-handle limit reached");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another instance operation is active");
  }
  signal = (struct sagr_signal *)calloc(1, sizeof(*signal));
  if (signal == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1,
                     errno, "could not allocate signal handle");
  }
  initialize_signal_request_for_handle(
      &request, SAGR_WIRE_SIGNAL_OPCODE_CREATE, NULL);
  request.value_bits = signal_value_to_bits(local_create.initial_value);
  instance->operation_active = 1;
  status = exchange_signal_request(
      instance, &request, &deadline, local_operation.cancel_fd, &response,
      &wire_status, &native_errno, &reason, NULL);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    free(signal);
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.signal_id == 0 ||
      response.signal_id > SAGR_SIGNAL_MAX_LIVE_SIGNALS ||
      response.generation == 0 ||
      response.generation <= instance->last_signal_generation ||
      response.sequence != 0 ||
      response.value_bits != request.value_bits || response.ready != 0 ||
      signal_id_is_active(instance, response.signal_id)) {
    poison_queue_transport(instance);
    free(signal);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "invalid successful signal CREATE ACK");
  }
  signal->magic = SAGR_SIGNAL_MAGIC;
  signal->instance = instance;
  signal->signal_id = response.signal_id;
  signal->generation = response.generation;
  signal->value = local_create.initial_value;
  signal->next_sequence = UINT64_C(1);
  signal->next = instance->signals;
  instance->signals = signal;
  ++instance->signal_count;
  instance->last_signal_generation = response.generation;
  if (out_info != NULL) {
    fill_signal_info(signal, out_info);
  }
  *out_signal = signal;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_signal_get_info(sagr_signal_t signal,
                                   sagr_signal_info_t *info,
                                   uint32_t info_size) {
  if (info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    memset(info, 0, info_size);
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(info, 0, sizeof(*info));
  if (!signal_handle_is_valid(signal)) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  fill_signal_info(signal, info);
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t signal_value_operation(
    sagr_signal_t signal, uint16_t opcode, int64_t store_value,
    const sagr_signal_operation_options_t *operation_options,
    int64_t *out_value, sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_signal_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_signal_request_t request;
  sagr_wire_signal_response_t response;
  struct sagr_instance *instance;
  struct sagr_signal candidate;
  sagr_status_t status;
  int native_errno = 0;
  int32_t wire_status = -1;
  int buffered_before;
  const char *reason = opcode == SAGR_WIRE_SIGNAL_OPCODE_LOAD
                           ? "signal load failed"
                           : "signal store failed";

  if (out_value != NULL) {
    *out_value = 0;
  }
  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (!signal_handle_is_valid(signal)) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid signal handle");
  }
  if (opcode == SAGR_WIRE_SIGNAL_OPCODE_LOAD && out_value == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "signal load output is null");
  }
  status = prepare_signal_operation(operation_options, &local_operation,
                                    &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid signal operation options");
  }
  instance = signal->instance;
  if (!signal_capability_selected(instance)) {
    return fail_open(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED, -1, 0,
                     "SIGNAL_EVENT_V1 was not negotiated");
  }
  status = require_signal_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE &&
      instance->pending_dispatch != NULL &&
      instance->pending_dispatch->signal == signal) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "signal participates in a pinned dispatch");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another instance operation is active");
  }
  initialize_signal_request_for_handle(&request, opcode, signal);
  if (opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE) {
    request.value_bits = signal_value_to_bits(store_value);
  }
  buffered_before = signal->completion_buffered;
  instance->operation_active = 1;
  status = exchange_signal_request(
      instance, &request, &deadline, local_operation.cancel_fd, &response,
      &wire_status, &native_errno, &reason, NULL);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    if (opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE &&
        signal->wait_pending != 0 && signal->wait_ready == 0 &&
        signal->wait_trigger_known == 0 && buffered_before == 0 &&
        signal->completion_buffered != 0) {
      poison_queue_transport(instance);
      return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                       wire_status, native_errno,
                       "completion accompanied a rejected signal STORE");
    }
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.signal_id != signal->signal_id ||
      response.generation != signal->generation || response.sequence != 0 ||
      response.ready != 0 ||
      (opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE &&
       response.value_bits != request.value_bits)) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0,
                     "invalid successful signal LOAD or STORE ACK");
  }
  candidate = *signal;
  candidate.value = signal_bits_to_value(response.value_bits);
  if (opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE && signal->wait_pending != 0 &&
      signal->wait_ready == 0 && signal->wait_trigger_known == 0) {
    if (signal_predicate_satisfied(signal->wait_condition,
                                   response.value_bits,
                                   signal->wait_compare_bits)) {
      if (response.sim_tick == UINT64_MAX) {
        poison_queue_transport(instance);
        return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                         wire_status, 0,
                         "successful STORE cannot trigger at maximum tick");
      }
      candidate.wait_trigger_known = 1;
      candidate.wait_trigger_value_bits = response.value_bits;
      candidate.wait_trigger_tick = response.sim_tick;
    } else if (buffered_before == 0 && signal->completion_buffered != 0) {
      poison_queue_transport(instance);
      return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                       wire_status, 0,
                       "signal completion preceded an unsatisfied STORE");
    }
  }
  if (signal->completion_buffered != 0 &&
      !signal_completion_is_canonical(&candidate, &signal->buffered_completion,
                                      0)) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0,
                     "buffered signal completion disagrees with STORE ACK");
  }
  signal->value = candidate.value;
  signal->wait_trigger_known = candidate.wait_trigger_known;
  signal->wait_trigger_value_bits = candidate.wait_trigger_value_bits;
  signal->wait_trigger_tick = candidate.wait_trigger_tick;
  if (out_value != NULL) {
    *out_value = signal->value;
  }
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_signal_load(
    sagr_signal_t signal,
    const sagr_signal_operation_options_t *operation_options,
    int64_t *out_value, sagr_error_info_t *out_error, uint32_t error_size) {
  return signal_value_operation(
      signal, SAGR_WIRE_SIGNAL_OPCODE_LOAD, 0, operation_options, out_value,
      out_error, error_size);
}

sagr_status_t sagr_signal_store(
    sagr_signal_t signal, int64_t value,
    const sagr_signal_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size) {
  return signal_value_operation(
      signal, SAGR_WIRE_SIGNAL_OPCODE_STORE, value, operation_options, NULL,
      out_error, error_size);
}

sagr_status_t sagr_signal_wait(
    sagr_signal_t signal, uint64_t condition, int64_t compare_value,
    const sagr_signal_operation_options_t *operation_options,
    sagr_signal_wait_result_t *out_result, uint32_t result_size,
    sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_signal_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_signal_request_t request;
  sagr_wire_signal_response_t response;
  sagr_wire_signal_response_t completion;
  struct sagr_instance *instance;
  sagr_status_t status;
  int native_errno = 0;
  int32_t wire_status = -1;
  uint64_t request_id = 0;
  const uint64_t compare_bits = signal_value_to_bits(compare_value);
  const char *reason = "signal wait failed";

  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_sized_output(out_result, result_size, sizeof(*out_result), 1);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid signal wait result buffer");
  }
  if (!signal_handle_is_valid(signal)) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid signal handle");
  }
  if (condition > SAGR_SIGNAL_CONDITION_GTE) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "invalid signal wait condition");
  }
  status = prepare_signal_operation(operation_options, &local_operation,
                                    &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid signal operation options");
  }
  instance = signal->instance;
  if (!signal_capability_selected(instance)) {
    return fail_open(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED, -1, 0,
                     "SIGNAL_EVENT_V1 was not negotiated");
  }
  status = require_signal_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (signal->wait_pending != 0 &&
      (signal->wait_condition != condition ||
       signal->wait_compare_bits != compare_bits)) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "signal already has a different pending wait");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another instance operation is active");
  }
  if (signal->wait_pending == 0) {
    if (instance->pending_signal_wait_count >=
            SAGR_SIGNAL_MAX_PENDING_WAITS ||
        signal->next_sequence == 0) {
      return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1,
                       0, "runtime pending signal-wait limit reached");
    }
    initialize_signal_request_for_handle(
        &request, SAGR_WIRE_SIGNAL_OPCODE_WAIT, signal);
    request.sequence = signal->next_sequence;
    request.value_bits = compare_bits;
    request.condition = condition;
    instance->operation_active = 1;
    status = exchange_signal_request(
        instance, &request, &deadline, local_operation.cancel_fd, &response,
        &wire_status, &native_errno, &reason, &request_id);
    instance->operation_active = 0;
    if (status != SAGR_STATUS_SUCCESS) {
      return fail_open(out_error, error_size, status, wire_status,
                       native_errno, reason);
    }
    if (response.signal_id != signal->signal_id ||
        response.generation != signal->generation ||
        response.sequence != request.sequence || response.ready > 1 ||
        response.sim_tick == UINT64_MAX ||
        response.ready !=
            (uint64_t)signal_predicate_satisfied(condition,
                                                  response.value_bits,
                                                  compare_bits)) {
      poison_queue_transport(instance);
      return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                       wire_status, 0, "invalid successful signal WAIT ACK");
    }
    signal->value = signal_bits_to_value(response.value_bits);
    signal->wait_pending = 1;
    signal->wait_sequence = request.sequence;
    signal->wait_condition = condition;
    signal->wait_compare_bits = compare_bits;
    signal->wait_request_id = request_id;
    signal->wait_ack_value_bits = response.value_bits;
    signal->wait_ack_tick = response.sim_tick;
    signal->wait_ready = response.ready;
    signal->next_sequence = request.sequence == UINT64_MAX
                                ? 0
                                : request.sequence + UINT64_C(1);
    ++instance->pending_signal_wait_count;
  }

  instance->operation_active = 1;
  memset(&completion, 0, sizeof(completion));
  status = receive_signal_completion(
      signal, &deadline, local_operation.cancel_fd, &completion, &wire_status,
      &native_errno, &reason);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    /* A canonical WAIT ACK may already have been admitted.  A subsequent
     * local deadline/cancellation is not a daemon response, so do not expose
     * the successful ACK status as if it were a failure ACK. */
    if (status == SAGR_STATUS_TIMED_OUT || status == SAGR_STATUS_CANCELLED) {
      wire_status = -1;
    }
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (instance->pending_signal_wait_count == 0) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_INTERNAL_ERROR,
                     wire_status, 0,
                     "pending signal-wait accounting underflow");
  }

  /* Publish the result only after the completion has passed every canonicality
   * check. Timeout, cancellation, and protocol failures leave only the
   * initialized struct_size in the caller's buffer. */
  out_result->status = SAGR_STATUS_SUCCESS;
  out_result->wire_status = wire_status;
  out_result->signal_id = signal->signal_id;
  out_result->generation = signal->generation;
  out_result->sequence = signal->wait_sequence;
  out_result->observed_value = signal_bits_to_value(completion.value_bits);
  out_result->admission_tick = signal->wait_ack_tick;
  out_result->completion_tick = completion.sim_tick;
  out_result->ready_at_admission = (uint32_t)signal->wait_ready;
  signal->value = signal_bits_to_value(completion.value_bits);
  if (instance->pending_dispatch != NULL &&
      instance->pending_dispatch->signal == signal &&
      completion.value_bits ==
          instance->pending_dispatch->expected_signal_value_bits) {
    instance->pending_dispatch->signal_completion_consumed = 1;
  }
  --instance->pending_signal_wait_count;
  signal->wait_pending = 0;
  signal->wait_sequence = 0;
  signal->wait_condition = 0;
  signal->wait_compare_bits = 0;
  signal->wait_request_id = 0;
  signal->wait_ack_value_bits = 0;
  signal->wait_ack_tick = 0;
  signal->wait_ready = 0;
  signal->wait_trigger_known = 0;
  signal->wait_trigger_value_bits = 0;
  signal->wait_trigger_tick = 0;
  release_consumed_dispatch(instance);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_signal_destroy(
    sagr_signal_t *signal,
    const sagr_signal_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_signal_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_signal_request_t request;
  sagr_wire_signal_response_t response;
  struct sagr_signal *owned_signal;
  struct sagr_signal **link;
  struct sagr_instance *instance;
  sagr_status_t status;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "signal destruction failed";

  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (signal == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "signal pointer is null");
  }
  if (*signal == NULL) {
    return SAGR_STATUS_SUCCESS;
  }
  owned_signal = *signal;
  if (!signal_handle_is_valid(owned_signal)) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid signal handle");
  }
  instance = owned_signal->instance;
  status = require_signal_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_signal_operation(operation_options, &local_operation,
                                    &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid signal operation options");
  }
  if (owned_signal->wait_pending != 0 ||
      owned_signal->completion_buffered != 0 ||
      (instance->pending_dispatch != NULL &&
       instance->pending_dispatch->signal == owned_signal)) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "signal has a pending wait completion");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another instance operation is active");
  }
  initialize_signal_request_for_handle(
      &request, SAGR_WIRE_SIGNAL_OPCODE_DESTROY, owned_signal);
  instance->operation_active = 1;
  status = exchange_signal_request(
      instance, &request, &deadline, local_operation.cancel_fd, &response,
      &wire_status, &native_errno, &reason, NULL);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.signal_id != owned_signal->signal_id ||
      response.generation != owned_signal->generation ||
      response.sequence != 0 || response.value_bits != 0 ||
      response.ready != 0) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0,
                     "invalid successful signal DESTROY ACK");
  }
  link = &instance->signals;
  while (*link != NULL && *link != owned_signal) {
    link = &(*link)->next;
  }
  if (*link != owned_signal || instance->signal_count == 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_INTERNAL_ERROR,
                     wire_status, 0, "signal ownership list is inconsistent");
  }
  *link = owned_signal->next;
  --instance->signal_count;
  *signal = NULL;
  memset(owned_signal, 0, sizeof(*owned_signal));
  free(owned_signal);
  return SAGR_STATUS_SUCCESS;
}

static void fill_dispatch_ticket(
    const struct sagr_pending_dispatch *pending,
    sagr_dispatch_ticket_t *ticket) {
  const sagr_wire_dispatch_response_t *ack = &pending->ack;
  ticket->struct_size = (uint32_t)sizeof(*ticket);
  ticket->request_id = pending->request_id;
  ticket->queue_id = ack->queue_id;
  ticket->queue_generation = ack->queue_generation;
  ticket->queue_sequence = ack->queue_sequence;
  ticket->fixture_id = ack->fixture_id;
  ticket->input_allocation_id = ack->input_allocation_id;
  ticket->input_generation = ack->input_generation;
  ticket->output_allocation_id = ack->output_allocation_id;
  ticket->output_generation = ack->output_generation;
  ticket->signal_id = ack->signal_id;
  ticket->signal_generation = ack->signal_generation;
  ticket->trace_id = ack->trace_id;
  ticket->input_gpu_va = ack->input_gpu_va;
  ticket->output_gpu_va = ack->output_gpu_va;
  ticket->packet_crc32c = ack->packet_crc32c;
  ticket->admission_tick = ack->admission_tick;
}

static int dispatch_ticket_matches(
    const struct sagr_pending_dispatch *pending,
    const sagr_dispatch_ticket_t *ticket) {
  const sagr_wire_dispatch_response_t *ack = &pending->ack;
  return ticket != NULL && ticket->struct_size >= sizeof(*ticket) &&
         ticket->flags == 0 && ticket->reserved0 == 0 &&
         reserved_is_zero(ticket->reserved, sizeof(ticket->reserved)) &&
         ticket->request_id == pending->request_id &&
         ticket->queue_id == ack->queue_id &&
         ticket->queue_generation == ack->queue_generation &&
         ticket->queue_sequence == ack->queue_sequence &&
         ticket->fixture_id == ack->fixture_id &&
         ticket->input_allocation_id == ack->input_allocation_id &&
         ticket->input_generation == ack->input_generation &&
         ticket->output_allocation_id == ack->output_allocation_id &&
         ticket->output_generation == ack->output_generation &&
         ticket->signal_id == ack->signal_id &&
         ticket->signal_generation == ack->signal_generation &&
         ticket->trace_id == ack->trace_id &&
         ticket->input_gpu_va == ack->input_gpu_va &&
         ticket->output_gpu_va == ack->output_gpu_va &&
         ticket->packet_crc32c == ack->packet_crc32c &&
         ticket->admission_tick == ack->admission_tick;
}

static void fill_dispatch_completion(
    const sagr_wire_dispatch_response_t *wire,
    sagr_dispatch_completion_t *completion, int32_t wire_status) {
  completion->struct_size = (uint32_t)sizeof(*completion);
  completion->status = SAGR_STATUS_SUCCESS;
  completion->wire_status = wire_status;
  completion->request_id = wire->request_id;
  completion->queue_id = wire->queue_id;
  completion->queue_generation = wire->queue_generation;
  completion->queue_sequence = wire->queue_sequence;
  completion->fixture_id = wire->fixture_id;
  completion->input_allocation_id = wire->input_allocation_id;
  completion->input_generation = wire->input_generation;
  completion->output_allocation_id = wire->output_allocation_id;
  completion->output_generation = wire->output_generation;
  completion->signal_id = wire->signal_id;
  completion->signal_generation = wire->signal_generation;
  completion->trace_id = wire->trace_id;
  completion->input_gpu_va = wire->input_gpu_va;
  completion->output_gpu_va = wire->output_gpu_va;
  completion->packet_crc32c = wire->packet_crc32c;
  completion->output_crc32c = wire->output_crc32c;
  completion->admission_tick = wire->admission_tick;
  completion->start_tick = wire->start_tick;
  completion->end_tick = wire->end_tick;
  completion->retire_tick = wire->retire_tick;
}

sagr_status_t sagr_queue_submit_pinned_dispatch(
    sagr_queue_t queue, sagr_memory_t input_memory,
    sagr_memory_t output_memory, sagr_signal_t completion_signal,
    const sagr_pinned_dispatch_options_t *options,
    const sagr_queue_operation_options_t *operation_options,
    sagr_dispatch_ticket_t *out_ticket, uint32_t ticket_size,
    sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_pinned_dispatch_options_t local_options;
  sagr_queue_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_dispatch_request_t request;
  sagr_wire_dispatch_response_t response;
  struct sagr_pending_dispatch *pending;
  struct sagr_instance *instance;
  sagr_status_t status;
  uint64_t request_id = 0;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "pinned dispatch submission failed";

  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_sized_output(out_ticket, ticket_size, sizeof(*out_ticket), 1);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid dispatch ticket output buffer");
  }
  if (queue == NULL || queue->magic != SAGR_QUEUE_MAGIC ||
      !memory_handle_is_valid(input_memory) ||
      !memory_handle_is_valid(output_memory) ||
      !signal_handle_is_valid(completion_signal)) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid dispatch resource handle");
  }
  instance = queue->instance;
  if (instance == NULL || instance->magic != SAGR_INSTANCE_MAGIC ||
      input_memory->instance != instance || output_memory->instance != instance ||
      completion_signal->instance != instance) {
    return fail_open(out_error, error_size, SAGR_STATUS_INSTANCE_MISMATCH, -1, 0,
                     "dispatch resources belong to different instances");
  }
  if (options == NULL) {
    status = sagr_pinned_dispatch_options_init(
        &local_options, (uint32_t)sizeof(local_options));
  } else if (options->struct_size < sizeof(*options)) {
    status = SAGR_STATUS_INVALID_ARGUMENT;
  } else {
    memcpy(&local_options, options, sizeof(local_options));
    status = validate_pinned_dispatch_options(&local_options);
  }
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid pinned dispatch options");
  }
  status = prepare_queue_operation(operation_options, &local_operation,
                                   &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid dispatch operation options");
  }
  if (!dispatch_capability_selected(instance) ||
      !queue_capability_selected(instance) ||
      !memory_capability_selected(instance) ||
      !signal_capability_selected(instance)) {
    return fail_open(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED, -1, 0,
                     "PINNED_DISPATCH_V1 prerequisites were not negotiated");
  }
  status = require_dispatch_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (instance->pending_dispatch != NULL || queue->pending_count != 0 ||
      queue->buffered_count != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "queue or session already has pending work");
  }
  if (input_memory == output_memory ||
      input_memory->allocation_id == output_memory->allocation_id ||
      input_memory->size_bytes < SAGR_DISPATCH_FIXED_IO_BYTES ||
      output_memory->size_bytes < SAGR_DISPATCH_FIXED_IO_BYTES ||
      input_memory->simulated_va == 0 || output_memory->simulated_va == 0 ||
      input_memory->simulated_va == output_memory->simulated_va) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "dispatch allocations are aliased or undersized");
  }
  if (completion_signal->value != INT64_C(1) ||
      completion_signal->wait_pending == 0 ||
      completion_signal->wait_condition != SAGR_SIGNAL_CONDITION_EQ ||
      completion_signal->wait_compare_bits != UINT64_C(0) ||
      completion_signal->wait_ready != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "dispatch signal must be signed one with an armed EQ-zero wait");
  }
  if (queue->next_sequence == 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1, 0,
                     "queue sequence space exhausted");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another instance operation is active");
  }
  pending = (struct sagr_pending_dispatch *)calloc(1, sizeof(*pending));
  if (pending == NULL) {
    return fail_open(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, -1,
                     errno, "could not allocate dispatch ticket state");
  }

  memset(&request, 0, sizeof(request));
  request.major = SAGR_DISPATCH_PROTOCOL_MAJOR;
  request.minor = SAGR_DISPATCH_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_DISPATCH_OPCODE_SUBMIT_PINNED;
  request.queue_id = queue->queue_id;
  request.queue_generation = queue->generation;
  request.queue_sequence = queue->next_sequence;
  request.fixture_id = local_options.fixture_id;
  request.input_allocation_id = input_memory->allocation_id;
  request.input_generation = input_memory->generation;
  request.output_allocation_id = output_memory->allocation_id;
  request.output_generation = output_memory->generation;
  request.signal_id = completion_signal->signal_id;
  request.signal_generation = completion_signal->generation;
  request.expected_signal_value_bits = UINT64_C(0);
  memcpy(request.fixture_manifest_sha256,
         sagr_dispatch_fixture_manifest_sha256,
         sizeof(request.fixture_manifest_sha256));
  instance->operation_active = 1;
  status = exchange_dispatch_request(
      instance, &request, &deadline, local_operation.cancel_fd, &response,
      &wire_status, &native_errno, &reason, &request_id);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    memset(pending, 0, sizeof(*pending));
    free(pending);
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (response.message_type != SAGR_WIRE_MESSAGE_DISPATCH_ACK ||
      response.opcode != request.opcode ||
      response.queue_id != request.queue_id ||
      response.queue_generation != request.queue_generation ||
      response.queue_sequence != request.queue_sequence ||
      response.fixture_id != request.fixture_id ||
      response.input_allocation_id != request.input_allocation_id ||
      response.input_generation != request.input_generation ||
      response.output_allocation_id != request.output_allocation_id ||
      response.output_generation != request.output_generation ||
      response.signal_id != request.signal_id ||
      response.signal_generation != request.signal_generation ||
      response.trace_id == 0 ||
      response.input_gpu_va != input_memory->simulated_va ||
      response.output_gpu_va != output_memory->simulated_va ||
      response.packet_crc32c != SAGR_DISPATCH_PACKET_CRC32C ||
      response.output_crc32c != 0 || response.start_tick != 0 ||
      response.end_tick != 0 || response.retire_tick != 0) {
    poison_queue_transport(instance);
    memset(pending, 0, sizeof(*pending));
    free(pending);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "invalid successful dispatch ACK");
  }
  pending->queue = queue;
  pending->input = input_memory;
  pending->output = output_memory;
  pending->signal = completion_signal;
  pending->request_id = request_id;
  pending->expected_signal_value_bits = request.expected_signal_value_bits;
  pending->ack = response;
  instance->pending_dispatch = pending;
  queue->next_sequence = request.queue_sequence == UINT64_MAX
                             ? 0
                             : request.queue_sequence + UINT64_C(1);
  fill_dispatch_ticket(pending, out_ticket);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_queue_wait_pinned_dispatch(
    sagr_queue_t queue, const sagr_dispatch_ticket_t *ticket,
    const sagr_queue_operation_options_t *operation_options,
    sagr_dispatch_completion_t *out_completion, uint32_t completion_size,
    sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_queue_operation_options_t local_operation;
  monotonic_deadline_t deadline;
  sagr_wire_dispatch_response_t completion;
  struct sagr_pending_dispatch *pending;
  struct sagr_instance *instance;
  sagr_status_t status;
  int native_errno = 0;
  int32_t wire_status = -1;
  const char *reason = "pinned dispatch completion failed";

  status = validate_error_output(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = prepare_sized_output(out_completion, completion_size,
                                sizeof(*out_completion), 1);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, 0,
                     "invalid dispatch completion output buffer");
  }
  if (queue == NULL || queue->magic != SAGR_QUEUE_MAGIC ||
      queue->instance == NULL ||
      queue->instance->magic != SAGR_INSTANCE_MAGIC) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, -1, 0,
                     "invalid dispatch queue handle");
  }
  instance = queue->instance;
  status = require_dispatch_transport(instance, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  pending = instance->pending_dispatch;
  if (pending == NULL || pending->queue != queue ||
      pending->dispatch_completion_consumed != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, -1, 0,
                     "dispatch ticket is not pending on this queue");
  }
  if (!dispatch_ticket_matches(pending, ticket)) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "a different pinned dispatch tuple is pending");
  }
  status = prepare_queue_operation(operation_options, &local_operation,
                                   &deadline, &native_errno);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, -1, native_errno,
                     "invalid dispatch wait options");
  }
  if (instance->operation_active != 0) {
    return fail_open(out_error, error_size, SAGR_STATUS_BUSY, -1, 0,
                     "another instance operation is active");
  }
  instance->operation_active = 1;
  memset(&completion, 0, sizeof(completion));
  status = receive_dispatch_completion(
      pending, &deadline, local_operation.cancel_fd, &completion,
      &wire_status, &native_errno, &reason);
  instance->operation_active = 0;
  if (status != SAGR_STATUS_SUCCESS) {
    return fail_open(out_error, error_size, status, wire_status, native_errno,
                     reason);
  }
  if (!dispatch_completion_is_canonical(pending, &completion)) {
    poison_queue_transport(instance);
    return fail_open(out_error, error_size, SAGR_STATUS_PROTOCOL_ERROR,
                     wire_status, 0, "dispatch completion result mismatch");
  }
  fill_dispatch_completion(&completion, out_completion, wire_status);
  pending->dispatch_completion_consumed = 1;
  observe_dispatch_signal_mirror(pending);
  release_consumed_dispatch(instance);
  return SAGR_STATUS_SUCCESS;
}
