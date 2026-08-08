/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <self_amdgpu_runtime/runtime.h>

#include "transport_internal.h"

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
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define SAGR_INSTANCE_MAGIC UINT64_C(0x53414752494e5354)

struct sagr_instance {
  uint64_t magic;
  int socket_fd;
  uint64_t next_request_id;
  sagr_instance_info_t info;
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
       SAGR_CAPABILITY_TOPOLOGY_MASK) == 0) {
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
                                 int *native_errno) {
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
      return SAGR_STATUS_SUCCESS;
    }
    if (count >= 0) {
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
                         local_options.cancel_fd, &native_errno);
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
  instance->next_request_id = request_id == UINT64_MAX ? 1 : request_id + 1;
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
  memset(&owned_instance->info, 0, sizeof(owned_instance->info));
  free(owned_instance);
  return SAGR_STATUS_SUCCESS;
}
