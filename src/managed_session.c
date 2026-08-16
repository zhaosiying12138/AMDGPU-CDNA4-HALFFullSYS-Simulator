/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <self_amdgpu_runtime/code_object.h>
#include <self_amdgpu_runtime/runtime.h>

#include "smi_registry_internal.h"
#include "managed_session_internal.h"
#include "managed_supervisor_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef SAGR_MANAGED_DEFAULT_PREFIX
#define SAGR_MANAGED_DEFAULT_PREFIX ""
#endif
#ifndef SAGR_MANAGED_DEFAULT_GEM5
#define SAGR_MANAGED_DEFAULT_GEM5 ""
#endif
#ifndef SAGR_MANAGED_DEFAULT_GEM5_CONFIG
#define SAGR_MANAGED_DEFAULT_GEM5_CONFIG ""
#endif
#ifndef SAGR_MANAGED_DEFAULT_REPO_ROOT
#define SAGR_MANAGED_DEFAULT_REPO_ROOT ""
#endif

#define SAGR_MANAGED_SESSION_MAGIC UINT64_C(0x534147524d534553)
#define SAGR_MANAGED_BUFFER_MAGIC UINT64_C(0x534147524d425546)
#define SAGR_MANAGED_KERNEL_MAGIC UINT64_C(0x534147524d4b4552)
#define SAGR_MANAGED_SUPERVISOR_ENV "SAGR_MANAGED_SUPERVISOR"
#define SAGR_MANAGED_SUPERVISOR_GRACE_MS UINT64_C(2000)

struct sagr_managed_buffer {
  uint64_t magic;
  struct sagr_managed_session *session;
  sagr_memory_t memory;
  sagr_memory_info_t info;
  struct sagr_managed_buffer *next;
};

struct sagr_managed_kernel {
  uint64_t magic;
  struct sagr_managed_session *session;
  sagr_generic_mapping_t mapping;
  sagr_generic_kernarg_t kernarg;
  sagr_generic_mapping_info_t mapping_info;
  sagr_generic_kernarg_info_t kernarg_info;
  sagr_code_object_kernel_info_t code_info;
  struct sagr_managed_kernel *next;
};

struct sagr_managed_session {
  uint64_t magic;
  sagr_managed_session_options_t options;
  sagr_instance_t instance;
  sagr_instance_info_t instance_info;
  sagr_queue_t queue;
  sagr_signal_t signal;
  struct sagr_managed_buffer *buffers;
  struct sagr_managed_kernel *kernels;
  pid_t child_pid;
  pid_t child_pgid;
  pid_t supervisor_pid;
  pid_t owner_pid;
  uint64_t epoch;
  uint32_t rank;
  uint32_t world_size;
  uint8_t job_uuid[SAGR_UUID_SIZE];
  char job_uuid_text[33];
  int exact_topology;
  int external_endpoint;
  int require_kmt;
  int supervised;
  char prefix[PATH_MAX];
  char gem5_path[PATH_MAX];
  char gem5_config_path[PATH_MAX];
  char supervisor_path[PATH_MAX];
  char repo_root[PATH_MAX];
  char run_dir[PATH_MAX];
  char endpoint[PATH_MAX];
  char trace_path[PATH_MAX];
  char output_dir[PATH_MAX];
  char log_path[PATH_MAX];
  sagr_smi_registry_lease_t smi_lease;
};

static int bytes_are_zero(const uint8_t *bytes, size_t size) {
  size_t index;
  for (index = 0U; index < size; ++index) {
    if (bytes[index] != 0U) {
      return 0;
    }
  }
  return 1;
}

static sagr_status_t prepare_error(sagr_error_info_t *error,
                                   uint32_t error_size) {
  if (error == NULL) {
    return error_size == 0U ? SAGR_STATUS_SUCCESS
                            : SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (error_size < sizeof(*error)) {
    if (error_size >= sizeof(error->struct_size)) {
      error->struct_size = (uint32_t)sizeof(*error);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(error, 0, error_size);
  error->struct_size = error_size;
  error->wire_status = -1;
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t fail(sagr_error_info_t *error, uint32_t error_size,
                          sagr_status_t status, int native_errno,
                          const char *message) {
  if (error != NULL && error_size >= sizeof(*error)) {
    error->status = status;
    error->wire_status = -1;
    error->native_errno = native_errno;
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return status;
}

static void succeed(sagr_error_info_t *error, uint32_t error_size,
                    const char *message) {
  if (error != NULL && error_size >= sizeof(*error)) {
    error->status = SAGR_STATUS_SUCCESS;
    error->wire_status = SAGR_STATUS_SUCCESS;
    error->native_errno = 0;
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
}

static int session_valid(const struct sagr_managed_session *session) {
  return session != NULL && session->magic == SAGR_MANAGED_SESSION_MAGIC &&
         session->instance != NULL && session->owner_pid == getpid();
}

static int buffer_valid(const struct sagr_managed_buffer *buffer) {
  return buffer != NULL && buffer->magic == SAGR_MANAGED_BUFFER_MAGIC &&
         session_valid(buffer->session) && buffer->memory != NULL;
}

static int kernel_valid(const struct sagr_managed_kernel *kernel) {
  return kernel != NULL && kernel->magic == SAGR_MANAGED_KERNEL_MAGIC &&
         session_valid(kernel->session) && kernel->mapping != NULL &&
         kernel->kernarg != NULL;
}

static uint64_t monotonic_milliseconds(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0U;
  }
  return (uint64_t)value.tv_sec * UINT64_C(1000) +
         (uint64_t)value.tv_nsec / UINT64_C(1000000);
}

static uint64_t timeout_milliseconds(uint64_t timeout_ns) {
  return timeout_ns / UINT64_C(1000000) +
         (timeout_ns % UINT64_C(1000000) != 0U ? UINT64_C(1) : UINT64_C(0));
}

static int copy_path(char destination[PATH_MAX], const char *value) {
  const size_t length = value != NULL ? strlen(value) : 0U;
  if (length == 0U || length >= PATH_MAX || value[0] != '/') {
    return 0;
  }
  memcpy(destination, value, length + 1U);
  return 1;
}

static int join_path(char destination[PATH_MAX], const char *left,
                     const char *right) {
  const int count = snprintf(destination, PATH_MAX, "%s/%s", left, right);
  return count > 0 && count < PATH_MAX;
}

static const char *selected_path(const char *managed_name,
                                 const char *opencl_name,
                                 const char *default_value) {
  const char *value = getenv(managed_name);
  if (value != NULL && value[0] != '\0') {
    return value;
  }
  value = getenv(opencl_name);
  return value != NULL && value[0] != '\0' ? value : default_value;
}

static int regular_file(const char *path, int executable) {
  struct stat state;
  return stat(path, &state) == 0 && S_ISREG(state.st_mode) &&
         (!executable || access(path, X_OK) == 0);
}

static int wait_for_pid(pid_t child, uint64_t timeout_ms, int *exit_code) {
  const uint64_t start = monotonic_milliseconds();
  struct timespec delay = {0, 10000000L};
  int status = 0;
  for (;;) {
    const pid_t result = waitpid(child, &status, WNOHANG);
    if (result == child) {
      if (exit_code != NULL) {
        *exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
      }
      return 1;
    }
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      return errno == ECHILD;
    }
    if (monotonic_milliseconds() - start >= timeout_ms) {
      return 0;
    }
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {
    }
    delay.tv_sec = 0;
    delay.tv_nsec = 10000000L;
  }
}

static void terminate_process_group(pid_t child) {
  int ignored = 0;
  if (child <= 0) {
    return;
  }
  (void)kill(-child, SIGTERM);
  if (!wait_for_pid(child, UINT64_C(2000), &ignored)) {
    (void)kill(-child, SIGKILL);
    (void)wait_for_pid(child, UINT64_C(2000), &ignored);
  }
}

static uint64_t mix_identity(uint64_t value) {
  value ^= value >> 30U;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27U;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

static void make_job_uuid(uint64_t epoch, uint64_t process_id,
                          uint8_t bytes[SAGR_UUID_SIZE], char text[33]) {
  static const char digits[] = "0123456789abcdef";
  uint64_t high = mix_identity(epoch ^ UINT64_C(0x6a09e667f3bcc909));
  uint64_t low = mix_identity(process_id ^ (epoch << 1U) ^
                              UINT64_C(0xbb67ae8584caa73b));
  size_t index;
  if (high == 0U && low == 0U) {
    low = UINT64_C(1);
  }
  for (index = 0U; index < 8U; ++index) {
    bytes[index] = (uint8_t)(high >> (index * 8U));
    bytes[index + 8U] = (uint8_t)(low >> (index * 8U));
  }
  bytes[6] = (uint8_t)((bytes[6] & 0x0fU) | 0x40U);
  bytes[8] = (uint8_t)((bytes[8] & 0x3fU) | 0x80U);
  for (index = 0U; index < SAGR_UUID_SIZE; ++index) {
    text[index * 2U] = digits[bytes[index] >> 4U];
    text[index * 2U + 1U] = digits[bytes[index] & 0x0fU];
  }
  text[32] = '\0';
}

static void format_job_uuid(const uint8_t bytes[SAGR_UUID_SIZE],
                            char text[33]) {
  static const char digits[] = "0123456789abcdef";
  size_t index;
  for (index = 0U; index < SAGR_UUID_SIZE; ++index) {
    text[index * 2U] = digits[bytes[index] >> 4U];
    text[index * 2U + 1U] = digits[bytes[index] & 0x0fU];
  }
  text[32] = '\0';
}

static uint64_t make_epoch(void) {
  struct timespec value;
  uint64_t result = (uint64_t)(uint32_t)getpid() << 32U;
  if (clock_gettime(CLOCK_MONOTONIC, &value) == 0) {
    result ^= (uint64_t)value.tv_sec;
    result ^= (uint64_t)value.tv_nsec << 1U;
  }
  result ^= (uint64_t)(uint32_t)getuid() << 17U;
  return result != 0U ? result : UINT64_C(1);
}

static int configure_spawn(posix_spawn_file_actions_t *actions,
                           posix_spawnattr_t *attributes,
                           const char *working_dir, const char *log_path,
                           int status_read_fd, int status_write_fd) {
  const short flags = POSIX_SPAWN_SETPGROUP;
  if (posix_spawn_file_actions_init(actions) != 0 ||
      posix_spawnattr_init(attributes) != 0 ||
      posix_spawnattr_setflags(attributes, flags) != 0 ||
      posix_spawnattr_setpgroup(attributes, 0) != 0 ||
      posix_spawn_file_actions_addopen(actions, STDIN_FILENO, "/dev/null",
                                       O_RDONLY, 0) != 0 ||
      posix_spawn_file_actions_addopen(actions, STDOUT_FILENO, log_path,
                                       O_WRONLY | O_CREAT | O_TRUNC,
                                       S_IRUSR | S_IWUSR) != 0 ||
      posix_spawn_file_actions_adddup2(actions, STDOUT_FILENO,
                                       STDERR_FILENO) != 0) {
    return 0;
  }
#if defined(__GLIBC__)
  if (working_dir != NULL && working_dir[0] != '\0' &&
      posix_spawn_file_actions_addchdir_np(actions, working_dir) != 0) {
    return 0;
  }
#else
  if (working_dir != NULL && working_dir[0] != '\0') {
    return 0;
  }
#endif
  if (status_read_fd >= 0 && status_write_fd >= 0 &&
      (posix_spawn_file_actions_addclose(actions, status_read_fd) != 0 ||
       posix_spawn_file_actions_adddup2(
           actions, status_write_fd, SAGR_MANAGED_SUPERVISOR_REPORT_FD) !=
           0 ||
       (status_write_fd != SAGR_MANAGED_SUPERVISOR_REPORT_FD &&
        posix_spawn_file_actions_addclose(actions, status_write_fd) != 0))) {
    return 0;
  }
  return 1;
}

static int read_supervisor_report(int descriptor, uint64_t timeout_ms,
                                  sagr_managed_supervisor_report_t *report) {
  const uint64_t start = monotonic_milliseconds();
  uint8_t *bytes = (uint8_t *)report;
  size_t offset = 0U;
  if (descriptor < 0 || report == NULL || timeout_ms == 0U) {
    errno = EINVAL;
    return 0;
  }
  memset(report, 0, sizeof(*report));
  while (offset < sizeof(*report)) {
    struct pollfd readiness;
    uint64_t elapsed = monotonic_milliseconds() - start;
    int remaining;
    ssize_t count;
    if (elapsed >= timeout_ms) {
      errno = ETIMEDOUT;
      return 0;
    }
    remaining = (int)(timeout_ms - elapsed > (uint64_t)INT_MAX
                          ? (uint64_t)INT_MAX
                          : timeout_ms - elapsed);
    memset(&readiness, 0, sizeof(readiness));
    readiness.fd = descriptor;
    readiness.events = POLLIN;
    do {
      count = poll(&readiness, 1U, remaining);
    } while (count < 0 && errno == EINTR);
    if (count <= 0) {
      if (count == 0) {
        errno = ETIMEDOUT;
      }
      return 0;
    }
    do {
      count = read(descriptor, bytes + offset, sizeof(*report) - offset);
    } while (count < 0 && errno == EINTR);
    if (count <= 0) {
      errno = count == 0 ? EPIPE : errno;
      return 0;
    }
    offset += (size_t)count;
  }
  if (report->magic != SAGR_MANAGED_SUPERVISOR_REPORT_MAGIC ||
      report->version != SAGR_MANAGED_SUPERVISOR_PROTOCOL_VERSION ||
      report->error_number != 0 || report->daemon_pid <= 0 ||
      report->daemon_pid > (int64_t)INT32_MAX ||
      !bytes_are_zero(report->reserved, sizeof(report->reserved))) {
    errno = report->error_number != 0 ? report->error_number : EPROTO;
    return 0;
  }
  return 1;
}

static int spawn_gem5(struct sagr_managed_session *session) {
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  char epoch[32];
  char rank[16];
  char world_size[16];
  char startup_timeout[32];
  char run_timeout[32];
  char owner_pid[32];
  char grace_ms[32];
  char status_fd[16];
  char path_environment[64];
  char home_environment[PATH_MAX + 16];
  char temp_environment[PATH_MAX + 16];
  char cache_environment[PATH_MAX + 32];
  char *environment[8];
  char *arguments[28];
  char *supervisor_arguments[40];
  sagr_managed_supervisor_report_t report;
  int report_pipe[2] = {-1, -1};
  pid_t spawned_pid = -1;
  int supervisor_index = 0;
  int index = 0;
  int count;
  if (!regular_file(session->gem5_path, 1) ||
      !regular_file(session->gem5_config_path, 0) ||
      access(session->repo_root, X_OK) != 0 ||
      (session->supervised && !regular_file(session->supervisor_path, 1))) {
    return 0;
  }
  count = snprintf(epoch, sizeof(epoch), "%llu",
                   (unsigned long long)session->epoch);
  if (count < 0 || (size_t)count >= sizeof(epoch) ||
      snprintf(rank, sizeof(rank), "%u", session->rank) < 0 ||
      snprintf(world_size, sizeof(world_size), "%u", session->world_size) <
          0 ||
      snprintf(startup_timeout, sizeof(startup_timeout), "%llu",
               (unsigned long long)timeout_milliseconds(
                   session->options.startup_timeout_ns)) < 0 ||
      snprintf(run_timeout, sizeof(run_timeout), "%llu",
               (unsigned long long)timeout_milliseconds(
                   session->options.run_timeout_ns)) < 0 ||
      snprintf(owner_pid, sizeof(owner_pid), "%ld", (long)getpid()) < 0 ||
      snprintf(grace_ms, sizeof(grace_ms), "%llu",
               (unsigned long long)SAGR_MANAGED_SUPERVISOR_GRACE_MS) < 0 ||
      snprintf(status_fd, sizeof(status_fd), "%d",
               SAGR_MANAGED_SUPERVISOR_REPORT_FD) < 0 ||
      snprintf(path_environment, sizeof(path_environment),
               "PATH=/usr/bin:/bin") < 0 ||
      snprintf(home_environment, sizeof(home_environment), "HOME=%s",
               session->run_dir) < 0 ||
      snprintf(temp_environment, sizeof(temp_environment), "TMPDIR=%s",
               session->run_dir) < 0 ||
      snprintf(cache_environment, sizeof(cache_environment),
               "XDG_CACHE_HOME=%s/cache", session->run_dir) < 0) {
    return 0;
  }
  environment[0] = path_environment;
  environment[1] = home_environment;
  environment[2] = temp_environment;
  environment[3] = cache_environment;
  environment[4] = (char *)"LC_ALL=C";
  environment[5] = (char *)"PYTHONNOUSERSITE=1";
  environment[6] = (char *)"PYTHONDONTWRITEBYTECODE=1";
  environment[7] = NULL;

  arguments[index++] = session->gem5_path;
  arguments[index++] = (char *)"--listener-mode=on";
  arguments[index++] = (char *)"--outdir";
  arguments[index++] = session->output_dir;
  arguments[index++] = session->gem5_config_path;
  arguments[index++] = (char *)"--endpoint";
  arguments[index++] = session->endpoint;
  arguments[index++] = (char *)"--dispatch-trace-path";
  arguments[index++] = session->trace_path;
  arguments[index++] = (char *)"--epoch";
  arguments[index++] = epoch;
  arguments[index++] = (char *)"--job-uuid";
  arguments[index++] = session->job_uuid_text;
  arguments[index++] = (char *)"--rank";
  arguments[index++] = rank;
  arguments[index++] = (char *)"--world-size";
  arguments[index++] = world_size;
  arguments[index++] = (char *)"--startup-timeout-ms";
  arguments[index++] = startup_timeout;
  arguments[index++] = (char *)"--handshake-timeout-ms";
  arguments[index++] = startup_timeout;
  arguments[index++] = (char *)"--run-timeout-ms";
  arguments[index++] = run_timeout;
  arguments[index] = NULL;

  if (session->supervised) {
    supervisor_arguments[supervisor_index++] = session->supervisor_path;
    supervisor_arguments[supervisor_index++] = (char *)"--owner-pid";
    supervisor_arguments[supervisor_index++] = owner_pid;
    supervisor_arguments[supervisor_index++] = (char *)"--status-fd";
    supervisor_arguments[supervisor_index++] = status_fd;
    supervisor_arguments[supervisor_index++] = (char *)"--grace-ms";
    supervisor_arguments[supervisor_index++] = grace_ms;
    supervisor_arguments[supervisor_index++] = (char *)"--";
    for (count = 0; count < index; ++count) {
      supervisor_arguments[supervisor_index++] = arguments[count];
    }
    supervisor_arguments[supervisor_index] = NULL;
    if (pipe2(report_pipe, O_CLOEXEC) != 0) {
      return 0;
    }
  }
  if (!configure_spawn(&actions, &attributes, session->repo_root,
                       session->log_path,
                       session->supervised ? report_pipe[0] : -1,
                       session->supervised ? report_pipe[1] : -1)) {
    goto cleanup;
  }
  if (posix_spawn(&spawned_pid,
                  session->supervised ? session->supervisor_path
                                      : session->gem5_path,
                  &actions, &attributes,
                  session->supervised ? supervisor_arguments : arguments,
                  environment) != 0) {
    spawned_pid = -1;
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  (void)posix_spawnattr_destroy(&attributes);
  if (session->supervised) {
    (void)close(report_pipe[1]);
    report_pipe[1] = -1;
  }
  if (spawned_pid <= 0) {
    goto cleanup;
  }
  if (!session->supervised) {
    session->child_pid = spawned_pid;
    session->child_pgid = spawned_pid;
    return 1;
  }
  session->supervisor_pid = spawned_pid;
  if (!read_supervisor_report(
          report_pipe[0], timeout_milliseconds(session->options.startup_timeout_ns),
          &report)) {
    terminate_process_group(session->supervisor_pid);
    session->supervisor_pid = -1;
    goto cleanup;
  }
  (void)close(report_pipe[0]);
  report_pipe[0] = -1;
  session->child_pid = (pid_t)report.daemon_pid;
  session->child_pgid = session->child_pid;
  return 1;

cleanup:
  if (report_pipe[0] >= 0) {
    (void)close(report_pipe[0]);
  }
  if (report_pipe[1] >= 0) {
    (void)close(report_pipe[1]);
  }
  if (spawned_pid > 0) {
    terminate_process_group(spawned_pid);
  }
  return 0;
}

static int wait_for_endpoint(struct sagr_managed_session *session) {
  const uint64_t start = monotonic_milliseconds();
  const uint64_t timeout =
      timeout_milliseconds(session->options.startup_timeout_ns);
  struct timespec delay = {0, 50000000L};
  for (;;) {
    struct stat state;
    int status = 0;
    const pid_t exited = waitpid(session->child_pid, &status, WNOHANG);
    if (exited == session->child_pid || (exited < 0 && errno != EINTR)) {
      session->child_pid = -1;
      session->child_pgid = -1;
      return 0;
    }
    if (lstat(session->endpoint, &state) == 0 && S_ISSOCK(state.st_mode) &&
        state.st_uid == getuid() &&
        (state.st_mode & (S_IRWXG | S_IRWXO)) == 0) {
      return 1;
    }
    if (monotonic_milliseconds() - start >= timeout) {
      return 0;
    }
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {
    }
    delay.tv_sec = 0;
    delay.tv_nsec = 50000000L;
  }
}

static int reserved_zero(const uint8_t *reserved, size_t size) {
  return bytes_are_zero(reserved, size);
}

static sagr_status_t validate_session_options(
    const sagr_managed_session_options_t *options) {
  const uint32_t known_flags = SAGR_MANAGED_SESSION_FLAG_KMT_PROVIDER;
  if (options == NULL || options->struct_size < sizeof(*options) ||
      options->version != SAGR_MANAGED_RUNTIME_API_VERSION ||
      (options->flags & ~known_flags) != 0U || options->queue_depth == 0U ||
      options->queue_depth > SAGR_QUEUE_MAX_DEPTH ||
      options->startup_timeout_ns == 0U ||
      options->operation_timeout_ns == 0U || options->run_timeout_ns == 0U ||
      !reserved_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_session_options_v2(
    const sagr_managed_session_options_v2_t *options) {
  const uint32_t known_flags =
      SAGR_MANAGED_SESSION_V2_FLAG_EXTERNAL_ENDPOINT |
      SAGR_MANAGED_SESSION_V2_FLAG_PRIVATE_NAMESPACE |
      SAGR_MANAGED_SESSION_V2_FLAG_KMT_PROVIDER;
  const int external =
      options != NULL &&
      (options->flags & SAGR_MANAGED_SESSION_V2_FLAG_EXTERNAL_ENDPOINT) != 0U;
  const int private_namespace =
      options != NULL &&
      (options->flags & SAGR_MANAGED_SESSION_V2_FLAG_PRIVATE_NAMESPACE) != 0U;
  if (options == NULL || options->struct_size < sizeof(*options) ||
      options->version != SAGR_MANAGED_SESSION_OPTIONS_V2_VERSION ||
      (options->flags & ~known_flags) != 0U || options->queue_depth == 0U ||
      options->queue_depth > SAGR_QUEUE_MAX_DEPTH ||
      options->startup_timeout_ns == 0U ||
      options->operation_timeout_ns == 0U || options->run_timeout_ns == 0U ||
      options->epoch == 0U || options->world_size == 0U ||
      options->world_size > SAGR_MANAGED_MAX_WORLD_SIZE ||
      options->rank >= options->world_size ||
      bytes_are_zero(options->job_uuid, sizeof(options->job_uuid)) ||
      !reserved_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (external == private_namespace) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options->endpoint[0] != (uint8_t)'/' ||
      memchr(options->endpoint, '\0', sizeof(options->endpoint)) == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_managed_session_options_init(
    sagr_managed_session_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->version = SAGR_MANAGED_RUNTIME_API_VERSION;
  options->queue_depth = SAGR_MANAGED_DEFAULT_QUEUE_DEPTH;
  options->startup_timeout_ns = SAGR_MANAGED_DEFAULT_STARTUP_TIMEOUT_NS;
  options->operation_timeout_ns = SAGR_MANAGED_DEFAULT_OPERATION_TIMEOUT_NS;
  options->run_timeout_ns = SAGR_MANAGED_DEFAULT_RUN_TIMEOUT_NS;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_managed_session_options_v2_init(
    sagr_managed_session_options_v2_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->version = SAGR_MANAGED_SESSION_OPTIONS_V2_VERSION;
  options->queue_depth = SAGR_MANAGED_DEFAULT_QUEUE_DEPTH;
  options->startup_timeout_ns = SAGR_MANAGED_DEFAULT_STARTUP_TIMEOUT_NS;
  options->operation_timeout_ns = SAGR_MANAGED_DEFAULT_OPERATION_TIMEOUT_NS;
  options->run_timeout_ns = SAGR_MANAGED_DEFAULT_RUN_TIMEOUT_NS;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_managed_launch_options_init(
    sagr_managed_launch_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->version = SAGR_MANAGED_RUNTIME_API_VERSION;
  options->grid_x = 64U;
  options->grid_y = 1U;
  options->grid_z = 1U;
  options->workgroup_x = 64U;
  options->workgroup_y = 1U;
  options->workgroup_z = 1U;
  options->num_warps = 1U;
  options->num_ctas = 1U;
  options->wavefront_size = 64U;
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t resolve_paths(struct sagr_managed_session *session,
                                   sagr_error_info_t *error,
                                   uint32_t error_size) {
  const char *endpoint = getenv("SAGR_GENERIC_BRIDGE_ENDPOINT");
  if (!copy_path(session->prefix,
                 selected_path("SAGR_MANAGED_PREFIX", "SAGR_OPENCL_PREFIX",
                               SAGR_MANAGED_DEFAULT_PREFIX))) {
    return fail(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "managed runtime prefix is not an absolute path");
  }
  if (session->external_endpoint) {
    return SAGR_STATUS_SUCCESS;
  }
  if (!session->exact_topology && endpoint != NULL && endpoint[0] != '\0') {
    if (!copy_path(session->endpoint, endpoint)) {
      return fail(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                  "managed external endpoint is not an absolute path");
    }
    session->external_endpoint = 1;
    return SAGR_STATUS_SUCCESS;
  }
  if (!copy_path(session->gem5_path,
                 selected_path("SAGR_MANAGED_GEM5", "SAGR_OPENCL_GEM5",
                               SAGR_MANAGED_DEFAULT_GEM5)) ||
      !copy_path(session->gem5_config_path,
                 selected_path("SAGR_MANAGED_GEM5_CONFIG",
                               "SAGR_OPENCL_GEM5_CONFIG",
                               SAGR_MANAGED_DEFAULT_GEM5_CONFIG)) ||
      !copy_path(session->repo_root,
                 selected_path("SAGR_MANAGED_REPO_ROOT",
                               "SAGR_OPENCL_REPO_ROOT",
                               SAGR_MANAGED_DEFAULT_REPO_ROOT))) {
    return fail(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "managed gem5 paths are not absolute");
  }
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t open_instance(struct sagr_managed_session *session,
                                   sagr_error_info_t *error,
                                   uint32_t error_size) {
  uint64_t capabilities =
      SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK |
      SAGR_CAPABILITY_MEMORY_MASK | SAGR_CAPABILITY_SIGNAL_MASK |
      SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK |
      SAGR_CAPABILITY_GENERIC_DISPATCH_MASK |
      SAGR_CAPABILITY_GENERIC_EXECUTION_MASK;
  sagr_instance_open_options_t options;
  sagr_status_t status = sagr_instance_open_options_init(
      &options, (uint32_t)sizeof(options));
  if (status != SAGR_STATUS_SUCCESS) {
    return fail(error, error_size, status, 0,
                "could not initialize managed handshake options");
  }
  if (session->require_kmt) {
    capabilities |= SAGR_CAPABILITY_KMT_MASK;
  }
  options.open_timeout_ns = session->options.startup_timeout_ns;
  options.offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |= capabilities;
  options.required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |= capabilities;
  if (session->exact_topology || !session->external_endpoint) {
    memcpy(options.expected_job_uuid, session->job_uuid,
           sizeof(options.expected_job_uuid));
    options.expected_epoch = session->epoch;
    options.expected_rank = session->rank;
    options.expected_world_size = session->world_size;
  }
  status = sagr_instance_open(session->endpoint, &options, &session->instance,
                              error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  status = sagr_instance_get_info(session->instance, &session->instance_info,
                                  (uint32_t)sizeof(session->instance_info));
  if (status != SAGR_STATUS_SUCCESS) {
    (void)sagr_instance_close(&session->instance);
    return fail(error, error_size, status, 0,
                "could not query managed simulator identity");
  }
  return SAGR_STATUS_SUCCESS;
}

static void fill_session_info(const struct sagr_managed_session *session,
                              sagr_managed_session_info_t *info) {
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->version = SAGR_MANAGED_RUNTIME_API_VERSION;
  info->external_endpoint = session->external_endpoint != 0 ? 1U : 0U;
  info->connection_id = session->instance_info.connection_id;
  info->epoch = session->instance_info.epoch;
  info->rank = session->instance_info.rank;
  info->world_size = session->instance_info.world_size;
  info->child_pid = session->child_pid > 0 ? (uint32_t)session->child_pid : 0U;
  memcpy(info->daemon_uuid, session->instance_info.daemon_uuid,
         sizeof(info->daemon_uuid));
  memcpy(info->job_uuid, session->instance_info.job_uuid,
         sizeof(info->job_uuid));
}

static int register_managed_daemon(struct sagr_managed_session *session) {
  sagr_smi_registry_identity_t identity;
  if (session->external_endpoint || session->child_pid <= 0) {
    return 1;
  }
  memset(&identity, 0, sizeof(identity));
  identity.owner_pid = getpid();
  identity.daemon_pid = session->child_pid;
  identity.epoch = session->instance_info.epoch;
  identity.connection_id = session->instance_info.connection_id;
  identity.rank = session->instance_info.rank;
  identity.world_size = session->instance_info.world_size;
  memcpy(identity.job_uuid, session->instance_info.job_uuid,
         sizeof(identity.job_uuid));
  memcpy(identity.daemon_uuid, session->instance_info.daemon_uuid,
         sizeof(identity.daemon_uuid));
  identity.endpoint = session->endpoint;
  identity.exact_topology = session->exact_topology;
  return sagr_smi_registry_claim(NULL, &identity, &session->smi_lease);
}

static sagr_status_t open_managed_session(
    const sagr_managed_session_options_t *options,
    const sagr_managed_session_options_v2_t *exact_options,
    sagr_managed_session_t *out_session,
    sagr_managed_session_info_t *out_info, uint32_t info_size,
    sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_managed_session_options_t defaults;
  struct sagr_managed_session *session = NULL;
  sagr_status_t status;
  char template_path[PATH_MAX];
  char cache_path[PATH_MAX];
  int count;
  if (out_session != NULL) {
    *out_session = NULL;
  }
  status = prepare_error(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (out_session == NULL || (out_info == NULL && info_size != 0U)) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "invalid managed session output");
  }
  if (out_info != NULL && info_size < sizeof(*out_info)) {
    if (info_size >= sizeof(out_info->struct_size)) {
      out_info->struct_size = (uint32_t)sizeof(*out_info);
    }
    return fail(out_error, error_size, SAGR_STATUS_BUFFER_TOO_SMALL, 0,
                "managed session info buffer is too small");
  }
  if (options == NULL) {
    (void)sagr_managed_session_options_init(&defaults,
                                            (uint32_t)sizeof(defaults));
    if (exact_options != NULL) {
      status = validate_session_options_v2(exact_options);
      if (status != SAGR_STATUS_SUCCESS) {
        return fail(out_error, error_size, status, 0,
                    "invalid exact-topology managed session options");
      }
      defaults.queue_depth = exact_options->queue_depth;
      defaults.startup_timeout_ns = exact_options->startup_timeout_ns;
      defaults.operation_timeout_ns = exact_options->operation_timeout_ns;
      defaults.run_timeout_ns = exact_options->run_timeout_ns;
    }
    options = &defaults;
  } else if (exact_options != NULL) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "managed session option versions cannot be mixed");
  }
  status = validate_session_options(options);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail(out_error, error_size, status, 0,
                "invalid managed session options");
  }
  session = (struct sagr_managed_session *)calloc(1, sizeof(*session));
  if (session == NULL) {
    return fail(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, errno,
                "could not allocate managed session");
  }
  session->magic = SAGR_MANAGED_SESSION_MAGIC;
  sagr_smi_registry_lease_init(&session->smi_lease);
  memcpy(&session->options, options, sizeof(session->options));
  session->child_pid = -1;
  session->child_pgid = -1;
  session->owner_pid = getpid();
  session->rank = exact_options != NULL ? exact_options->rank : 0U;
  session->world_size =
      exact_options != NULL ? exact_options->world_size : 1U;
  session->require_kmt =
      exact_options != NULL
          ? (exact_options->flags &
             SAGR_MANAGED_SESSION_V2_FLAG_KMT_PROVIDER) != 0U
          : (options->flags & SAGR_MANAGED_SESSION_FLAG_KMT_PROVIDER) != 0U;
  if (exact_options != NULL) {
    session->exact_topology = 1;
    session->epoch = exact_options->epoch;
    memcpy(session->job_uuid, exact_options->job_uuid,
           sizeof(session->job_uuid));
    format_job_uuid(session->job_uuid, session->job_uuid_text);
    if ((exact_options->flags &
         SAGR_MANAGED_SESSION_V2_FLAG_EXTERNAL_ENDPOINT) != 0U) {
      if (!copy_path(session->endpoint,
                     (const char *)exact_options->endpoint)) {
        status = fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                      "exact-topology external endpoint is invalid");
        goto failure;
      }
      session->external_endpoint = 1;
    } else if (!copy_path(session->endpoint,
                          (const char *)exact_options->endpoint)) {
      status = fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                    "exact-topology private endpoint is invalid");
      goto failure;
    }
  }
  status = resolve_paths(session, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    goto failure;
  }
  if (!session->external_endpoint) {
    if (session->exact_topology) {
      const char *separator = strrchr(session->endpoint, '/');
      if (separator == NULL || separator == session->endpoint ||
          (size_t)(separator - session->endpoint) >= sizeof(session->run_dir)) {
        status = fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                      "exact-topology endpoint parent is invalid");
        goto failure;
      }
      memcpy(session->run_dir, session->endpoint,
             (size_t)(separator - session->endpoint));
      session->run_dir[separator - session->endpoint] = '\0';
      if (mkdir(session->run_dir, S_IRWXU) != 0 ||
          chmod(session->run_dir, S_IRWXU) != 0) {
        status = fail(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES,
                      errno, "could not create exact-topology run directory");
        goto failure;
      }
    } else {
      count = snprintf(template_path, sizeof(template_path),
                       "/tmp/self-amdgpu-opencl-run.%lu.XXXXXX",
                       (unsigned long)getuid());
      if (count < 0 || (size_t)count >= sizeof(template_path) ||
          mkdtemp(template_path) == NULL ||
          chmod(template_path, S_IRWXU) != 0 ||
          !copy_path(session->run_dir, template_path) ||
          !join_path(session->endpoint, session->run_dir, "bridge.sock")) {
        status = fail(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES,
                      errno,
                      "could not create managed simulator run directory");
        goto failure;
      }
    }
    if (
        !join_path(session->trace_path, session->run_dir,
                   "dispatch-trace.jsonl") ||
        !join_path(session->output_dir, session->run_dir, "m5out") ||
        !join_path(session->log_path, session->run_dir, "gem5.log") ||
        !join_path(cache_path, session->run_dir, "cache") ||
        mkdir(session->output_dir, S_IRWXU) != 0 ||
        mkdir(cache_path, S_IRWXU) != 0) {
      status = fail(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, errno,
                    "could not create managed simulator run directory");
      goto failure;
    }
    if (!session->exact_topology) {
      session->epoch = make_epoch();
      make_job_uuid(session->epoch, (uint64_t)(uint32_t)getpid(),
                    session->job_uuid, session->job_uuid_text);
    }
    if (!spawn_gem5(session) || !wait_for_endpoint(session)) {
      status = fail(out_error, error_size, SAGR_STATUS_UNAVAILABLE, errno,
                    "managed gem5 did not publish its private endpoint");
      goto failure;
    }
  }
  status = open_instance(session, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    goto failure;
  }
  if (!register_managed_daemon(session)) {
    status = fail(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, errno,
                  "managed simulator device registry has no free slot");
    goto failure;
  }
  if (out_info != NULL) {
    fill_session_info(session, out_info);
  }
  *out_session = session;
  succeed(out_error, error_size, "managed simulator session is ready");
  return SAGR_STATUS_SUCCESS;

failure:
  if (session->instance != NULL) {
    (void)sagr_instance_close(&session->instance);
  }
  if (!session->external_endpoint && session->child_pid > 0) {
    terminate_process_group(session->child_pid);
  }
  sagr_smi_registry_release(&session->smi_lease);
  session->magic = 0U;
  free(session);
  return status;
}

sagr_status_t sagr_managed_session_open(
    const sagr_managed_session_options_t *options,
    sagr_managed_session_t *out_session,
    sagr_managed_session_info_t *out_info, uint32_t info_size,
    sagr_error_info_t *out_error, uint32_t error_size) {
  return open_managed_session(options, NULL, out_session, out_info, info_size,
                              out_error, error_size);
}

sagr_status_t sagr_managed_session_open_v2(
    const sagr_managed_session_options_v2_t *options,
    sagr_managed_session_t *out_session,
    sagr_managed_session_info_t *out_info, uint32_t info_size,
    sagr_error_info_t *out_error, uint32_t error_size) {
  return open_managed_session(NULL, options, out_session, out_info, info_size,
                              out_error, error_size);
}

sagr_status_t sagr_managed_session_get_info(
    sagr_managed_session_t opaque_session,
    sagr_managed_session_info_t *out_info, uint32_t info_size) {
  struct sagr_managed_session *session =
      (struct sagr_managed_session *)opaque_session;
  if (!session_valid(session) || out_info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*out_info)) {
    if (info_size >= sizeof(out_info->struct_size)) {
      out_info->struct_size = (uint32_t)sizeof(*out_info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  fill_session_info(session, out_info);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_managed_session_get_instance(
    sagr_managed_session_t opaque_session, sagr_instance_t *out_instance) {
  struct sagr_managed_session *session =
      (struct sagr_managed_session *)opaque_session;
  if (out_instance != NULL) {
    *out_instance = NULL;
  }
  if (!session_valid(session) || out_instance == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  *out_instance = session->instance;
  return SAGR_STATUS_SUCCESS;
}

static void queue_operation(const struct sagr_managed_session *session,
                            sagr_queue_operation_options_t *operation) {
  (void)sagr_queue_operation_options_init(operation,
                                           (uint32_t)sizeof(*operation));
  operation->timeout_ns = session->options.operation_timeout_ns;
}

static void run_operation(const struct sagr_managed_session *session,
                          sagr_queue_operation_options_t *operation) {
  (void)sagr_queue_operation_options_init(operation,
                                           (uint32_t)sizeof(*operation));
  operation->timeout_ns = session->options.run_timeout_ns;
}

static void memory_operation(const struct sagr_managed_session *session,
                             sagr_memory_operation_options_t *operation) {
  (void)sagr_memory_operation_options_init(operation,
                                            (uint32_t)sizeof(*operation));
  operation->timeout_ns = session->options.operation_timeout_ns;
}

static void signal_operation(const struct sagr_managed_session *session,
                             sagr_signal_operation_options_t *operation) {
  (void)sagr_signal_operation_options_init(operation,
                                            (uint32_t)sizeof(*operation));
  operation->timeout_ns = session->options.operation_timeout_ns;
}

static sagr_status_t ensure_launch_resources(
    struct sagr_managed_session *session, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_queue_create_options_t queue_options;
  sagr_queue_operation_options_t queue_op;
  sagr_signal_create_options_t signal_options;
  sagr_signal_operation_options_t signal_op;
  sagr_status_t status;
  if (session->queue == NULL) {
    (void)sagr_queue_create_options_init(&queue_options,
                                         (uint32_t)sizeof(queue_options));
    queue_options.depth = session->options.queue_depth;
    queue_operation(session, &queue_op);
    status = sagr_queue_create(session->instance, &queue_options, &queue_op,
                               &session->queue, NULL, 0U, error, error_size);
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
  }
  if (session->signal == NULL) {
    (void)sagr_signal_create_options_init(
        &signal_options, (uint32_t)sizeof(signal_options));
    signal_options.initial_value = INT64_C(1);
    signal_operation(session, &signal_op);
    status = sagr_signal_create(session->instance, &signal_options, &signal_op,
                                &session->signal, NULL, 0U, error, error_size);
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
  }
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_managed_buffer_allocate(
    sagr_managed_session_t opaque_session, uint64_t size_bytes,
    uint64_t alignment_bytes, sagr_managed_buffer_t *out_buffer,
    sagr_memory_info_t *out_info, uint32_t info_size,
    sagr_error_info_t *out_error, uint32_t error_size) {
  struct sagr_managed_session *session =
      (struct sagr_managed_session *)opaque_session;
  struct sagr_managed_buffer *buffer = NULL;
  sagr_memory_allocate_options_t options;
  sagr_memory_operation_options_t operation;
  sagr_status_t status;
  if (out_buffer != NULL) {
    *out_buffer = NULL;
  }
  status = prepare_error(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (!session_valid(session) || out_buffer == NULL || size_bytes == 0U ||
      size_bytes > SAGR_MEMORY_MAX_SINGLE_ALLOCATION_BYTES ||
      (out_info == NULL && info_size != 0U)) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "invalid managed buffer allocation");
  }
  if (out_info != NULL && info_size < sizeof(*out_info)) {
    if (info_size >= sizeof(out_info->struct_size)) {
      out_info->struct_size = (uint32_t)sizeof(*out_info);
    }
    return fail(out_error, error_size, SAGR_STATUS_BUFFER_TOO_SMALL, 0,
                "managed buffer info buffer is too small");
  }
  buffer = (struct sagr_managed_buffer *)calloc(1, sizeof(*buffer));
  if (buffer == NULL) {
    return fail(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, errno,
                "could not allocate managed buffer handle");
  }
  (void)sagr_memory_allocate_options_init(&options,
                                           (uint32_t)sizeof(options));
  options.size_bytes = size_bytes;
  options.alignment_bytes = alignment_bytes == 0U
                                ? SAGR_MEMORY_ALIGNMENT_4K
                                : alignment_bytes;
  memory_operation(session, &operation);
  status = sagr_memory_allocate(session->instance, &options, &operation,
                                &buffer->memory, &buffer->info,
                                (uint32_t)sizeof(buffer->info), out_error,
                                error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    free(buffer);
    return status;
  }
  buffer->magic = SAGR_MANAGED_BUFFER_MAGIC;
  buffer->session = session;
  buffer->next = session->buffers;
  session->buffers = buffer;
  if (out_info != NULL) {
    memcpy(out_info, &buffer->info, sizeof(*out_info));
  }
  *out_buffer = buffer;
  succeed(out_error, error_size, "managed simulator buffer allocated");
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_managed_buffer_get_info(
    sagr_managed_buffer_t opaque_buffer, sagr_memory_info_t *out_info,
    uint32_t info_size) {
  struct sagr_managed_buffer *buffer =
      (struct sagr_managed_buffer *)opaque_buffer;
  if (!buffer_valid(buffer) || out_info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*out_info)) {
    if (info_size >= sizeof(out_info->struct_size)) {
      out_info->struct_size = (uint32_t)sizeof(*out_info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memcpy(out_info, &buffer->info, sizeof(*out_info));
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t buffer_copy(struct sagr_managed_buffer *buffer,
                                 uint64_t offset, void *bytes,
                                 uint64_t byte_count, int to_host,
                                 sagr_error_info_t *error,
                                 uint32_t error_size) {
  sagr_memory_operation_options_t operation;
  uint64_t copied = 0U;
  sagr_status_t status;
  if (!buffer_valid(buffer) || bytes == NULL || byte_count == 0U ||
      offset > buffer->info.size_bytes ||
      byte_count > buffer->info.size_bytes - offset) {
    return fail(error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "invalid managed buffer copy range");
  }
  memory_operation(buffer->session, &operation);
  while (copied < byte_count) {
    const uint64_t remaining = byte_count - copied;
    const uint64_t chunk = remaining > SAGR_MEMORY_MAX_TRANSFER_BYTES
                               ? SAGR_MEMORY_MAX_TRANSFER_BYTES
                               : remaining;
    uint8_t *host_bytes = (uint8_t *)bytes + (size_t)copied;
    status = to_host != 0
                 ? sagr_memory_copy_to_host(buffer->memory, offset + copied,
                                            host_bytes, chunk, &operation, error,
                                            error_size)
                 : sagr_memory_copy_from_host(buffer->memory, offset + copied,
                                              host_bytes, chunk, &operation,
                                              error, error_size);
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
    copied += chunk;
  }
  succeed(error, error_size,
          to_host != 0 ? "managed D2H copy completed"
                       : "managed H2D copy completed");
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_managed_buffer_copy_from_host(
    sagr_managed_buffer_t opaque_buffer, uint64_t offset, const void *source,
    uint64_t byte_count, sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_status_t status = prepare_error(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  return buffer_copy((struct sagr_managed_buffer *)opaque_buffer, offset,
                     (void *)source, byte_count, 0, out_error, error_size);
}

sagr_status_t sagr_managed_buffer_copy_to_host(
    sagr_managed_buffer_t opaque_buffer, uint64_t offset, void *destination,
    uint64_t byte_count, sagr_error_info_t *out_error, uint32_t error_size) {
  sagr_status_t status = prepare_error(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  return buffer_copy((struct sagr_managed_buffer *)opaque_buffer, offset,
                     destination, byte_count, 1, out_error, error_size);
}

sagr_status_t sagr_managed_buffer_free(
    sagr_managed_buffer_t *opaque_buffer, sagr_error_info_t *out_error,
    uint32_t error_size) {
  struct sagr_managed_buffer *buffer;
  struct sagr_managed_buffer **link;
  sagr_memory_operation_options_t operation;
  sagr_status_t status = prepare_error(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (opaque_buffer == NULL) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "managed buffer pointer is null");
  }
  if (*opaque_buffer == NULL) {
    return SAGR_STATUS_SUCCESS;
  }
  buffer = (struct sagr_managed_buffer *)*opaque_buffer;
  if (!buffer_valid(buffer)) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, 0,
                "invalid managed buffer handle");
  }
  memory_operation(buffer->session, &operation);
  status = sagr_memory_free(&buffer->memory, &operation, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  link = &buffer->session->buffers;
  while (*link != NULL && *link != buffer) {
    link = &(*link)->next;
  }
  if (*link != buffer) {
    return fail(out_error, error_size, SAGR_STATUS_INTERNAL_ERROR, 0,
                "managed buffer ownership list is inconsistent");
  }
  *link = buffer->next;
  buffer->magic = 0U;
  *opaque_buffer = NULL;
  free(buffer);
  succeed(out_error, error_size, "managed simulator buffer freed");
  return SAGR_STATUS_SUCCESS;
}

static void fill_kernel_info(const struct sagr_managed_kernel *kernel,
                             sagr_managed_kernel_info_t *info) {
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->version = SAGR_MANAGED_RUNTIME_API_VERSION;
  info->kernel_index = kernel->code_info.index;
  info->kernarg_segment_size = kernel->code_info.kernarg_segment_size;
  info->kernarg_segment_align = kernel->code_info.kernarg_segment_align;
  info->group_segment_fixed_size =
      kernel->code_info.group_segment_fixed_size;
  info->private_segment_fixed_size =
      kernel->code_info.private_segment_fixed_size;
  info->max_flat_workgroup_size = kernel->code_info.max_flat_workgroup_size;
  info->wavefront_size = kernel->code_info.wavefront_size;
  info->descriptor_preload_dwords =
      kernel->code_info.descriptor_kernarg_preload;
  info->object_id = kernel->mapping_info.object_id;
  info->object_generation = kernel->mapping_info.object_generation;
  info->mapping_id = kernel->mapping_info.mapping_id;
  info->mapping_generation = kernel->mapping_info.mapping_generation;
  info->entry_va = kernel->mapping_info.entry_va;
  info->kernarg_va = kernel->kernarg_info.kernarg_va;
  memcpy(info->image_sha256, kernel->mapping_info.image_sha256,
         sizeof(info->image_sha256));
  info->connection_id = kernel->mapping_info.connection_id;
  info->epoch = kernel->mapping_info.epoch;
  memcpy(info->daemon_uuid, kernel->mapping_info.daemon_uuid,
         sizeof(info->daemon_uuid));
}

sagr_status_t sagr_managed_kernel_load(
    sagr_managed_session_t opaque_session, const void *image,
    uint64_t image_size, const char *kernel_name,
    sagr_managed_kernel_t *out_kernel, sagr_managed_kernel_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size) {
  struct sagr_managed_session *session =
      (struct sagr_managed_session *)opaque_session;
  struct sagr_managed_kernel *kernel = NULL;
  sagr_code_object_info_t code_object;
  sagr_code_object_remote_info_t remote;
  sagr_generic_map_options_t map_options;
  sagr_generic_kernarg_allocate_options_t kernarg_options;
  sagr_queue_operation_options_t operation;
  sagr_status_t status;
  if (out_kernel != NULL) {
    *out_kernel = NULL;
  }
  status = prepare_error(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (!session_valid(session) || image == NULL || image_size == 0U ||
      image_size > SAGR_CODE_OBJECT_TRANSPORT_MAX_IMAGE_BYTES ||
      image_size > (uint64_t)SIZE_MAX || kernel_name == NULL ||
      kernel_name[0] == '\0' || out_kernel == NULL ||
      (out_info == NULL && info_size != 0U)) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "invalid managed kernel image or output");
  }
  if (out_info != NULL && info_size < sizeof(*out_info)) {
    if (info_size >= sizeof(out_info->struct_size)) {
      out_info->struct_size = (uint32_t)sizeof(*out_info);
    }
    return fail(out_error, error_size, SAGR_STATUS_BUFFER_TOO_SMALL, 0,
                "managed kernel info buffer is too small");
  }
  kernel = (struct sagr_managed_kernel *)calloc(1, sizeof(*kernel));
  if (kernel == NULL) {
    return fail(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES, errno,
                "could not allocate managed kernel handle");
  }
  memset(&code_object, 0, sizeof(code_object));
  status = sagr_code_object_validate(image, (size_t)image_size, &code_object,
                                     (uint32_t)sizeof(code_object));
  if (status != SAGR_STATUS_SUCCESS) {
    status = fail(out_error, error_size, status, 0,
                  "managed kernel code object validation failed");
    goto failure;
  }
  status = sagr_code_object_get_kernel(
      &code_object, kernel_name, &kernel->code_info,
      (uint32_t)sizeof(kernel->code_info));
  if (status != SAGR_STATUS_SUCCESS) {
    status = fail(out_error, error_size, status, 0,
                  "managed kernel metadata was not found");
    goto failure;
  }
  if (code_object.gfx_target != SAGR_CODE_OBJECT_TARGET_GFX950 ||
      kernel->code_info.relocation_count != 0U ||
      kernel->code_info.kernarg_segment_size == 0U ||
      kernel->code_info.kernarg_segment_size >
          SAGR_GENERIC_MAX_KERNARG_BYTES ||
      kernel->code_info.descriptor_kernarg_preload >
          SAGR_GENERIC_MAX_PRELOAD_DWORDS) {
    status = fail(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED, 0,
                  "managed kernel metadata is outside the gfx950 subset");
    goto failure;
  }
  queue_operation(session, &operation);
  memset(&remote, 0, sizeof(remote));
  status = sagr_code_object_upload(
      session->instance, image, (size_t)image_size, kernel_name, &operation,
      &remote, (uint32_t)sizeof(remote), out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    goto failure;
  }
  (void)sagr_generic_map_options_init(&map_options,
                                      (uint32_t)sizeof(map_options));
  map_options.object_id = remote.object_id;
  map_options.object_generation = remote.generation;
  map_options.kernel_index = kernel->code_info.index;
  map_options.gfx_target = code_object.gfx_target;
  map_options.relocation_count = kernel->code_info.relocation_count;
  map_options.kernarg_segment_size = kernel->code_info.kernarg_segment_size;
  map_options.kernarg_segment_align = kernel->code_info.kernarg_segment_align;
  map_options.descriptor_preload_dwords =
      kernel->code_info.descriptor_kernarg_preload;
  memcpy(map_options.image_sha256, remote.image_sha256,
         sizeof(map_options.image_sha256));
  if (strlen(kernel->code_info.name) >= sizeof(map_options.kernel_name)) {
    status = fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                  "managed kernel name exceeds the transport field");
    goto failure;
  }
  memcpy(map_options.kernel_name, kernel->code_info.name,
         strlen(kernel->code_info.name) + 1U);
  status = sagr_generic_map_object(
      session->instance, &map_options, &operation, &kernel->mapping,
      &kernel->mapping_info, (uint32_t)sizeof(kernel->mapping_info), out_error,
      error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    goto failure;
  }
  (void)sagr_generic_kernarg_allocate_options_init(
      &kernarg_options, (uint32_t)sizeof(kernarg_options));
  kernarg_options.size_bytes = kernel->code_info.kernarg_segment_size;
  kernarg_options.alignment_bytes =
      kernel->code_info.kernarg_segment_align < 8U
          ? UINT64_C(8)
          : (uint64_t)kernel->code_info.kernarg_segment_align;
  status = sagr_generic_alloc_kernarg(
      kernel->mapping, &kernarg_options, &operation, &kernel->kernarg,
      &kernel->kernarg_info, (uint32_t)sizeof(kernel->kernarg_info), out_error,
      error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    goto failure;
  }
  kernel->magic = SAGR_MANAGED_KERNEL_MAGIC;
  kernel->session = session;
  kernel->next = session->kernels;
  session->kernels = kernel;
  if (out_info != NULL) {
    fill_kernel_info(kernel, out_info);
  }
  *out_kernel = kernel;
  succeed(out_error, error_size, "managed kernel uploaded and mapped");
  return SAGR_STATUS_SUCCESS;

failure:
  if (kernel->mapping != NULL) {
    (void)sagr_generic_unmap_object(&kernel->mapping, &operation, NULL, 0U);
  }
  free(kernel);
  return status;
}

sagr_status_t sagr_managed_kernel_get_info(
    sagr_managed_kernel_t opaque_kernel,
    sagr_managed_kernel_info_t *out_info, uint32_t info_size) {
  struct sagr_managed_kernel *kernel =
      (struct sagr_managed_kernel *)opaque_kernel;
  if (!kernel_valid(kernel) || out_info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*out_info)) {
    if (info_size >= sizeof(out_info->struct_size)) {
      out_info->struct_size = (uint32_t)sizeof(*out_info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  fill_kernel_info(kernel, out_info);
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t validate_launch_options(
    const sagr_managed_launch_options_t *options) {
  uint64_t workgroup_size;
  uint64_t expected_size;
  if (options == NULL || options->struct_size < sizeof(*options) ||
      options->version != SAGR_MANAGED_RUNTIME_API_VERSION ||
      options->flags != 0U || options->grid_x == 0U ||
      options->grid_y == 0U || options->grid_z == 0U ||
      options->workgroup_x == 0U || options->workgroup_y == 0U ||
      options->workgroup_z == 0U ||
      options->workgroup_x > SAGR_GENERIC_MAX_WORKGROUP_DIMENSION ||
      options->workgroup_y > SAGR_GENERIC_MAX_WORKGROUP_DIMENSION ||
      options->workgroup_z > SAGR_GENERIC_MAX_WORKGROUP_DIMENSION ||
      options->grid_x < options->workgroup_x ||
      options->grid_y < options->workgroup_y ||
      options->grid_z < options->workgroup_z || options->num_warps == 0U ||
      options->num_warps > SAGR_GENERIC_MAX_WARPS ||
      options->num_ctas == 0U || options->num_ctas > SAGR_GENERIC_MAX_CTAS ||
      options->shared_memory_bytes > SAGR_GENERIC_MAX_SHARED_BYTES ||
      options->wavefront_size != 64U || options->launch_flags != 0U ||
      options->reserved0 != 0U ||
      !reserved_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  workgroup_size = (uint64_t)options->workgroup_x * options->workgroup_y *
                   options->workgroup_z;
  expected_size = (uint64_t)options->num_warps * options->wavefront_size;
  return workgroup_size <= SAGR_GENERIC_MAX_WORKGROUP_DIMENSION &&
                 workgroup_size == expected_size
             ? SAGR_STATUS_SUCCESS
             : SAGR_STATUS_INVALID_ARGUMENT;
}

sagr_status_t sagr_managed_kernel_launch(
    sagr_managed_kernel_t opaque_kernel, const void *packed_kernarg,
    uint64_t kernarg_size, const sagr_managed_launch_options_t *options,
    sagr_generic_dispatch_completion_t *out_completion,
    uint32_t completion_size, sagr_error_info_t *out_error,
    uint32_t error_size) {
  struct sagr_managed_kernel *kernel =
      (struct sagr_managed_kernel *)opaque_kernel;
  struct sagr_managed_session *session;
  sagr_generic_submit_options_t submit;
  sagr_generic_dispatch_ticket_t ticket;
  sagr_queue_operation_options_t operation;
  sagr_status_t status = prepare_error(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (!kernel_valid(kernel) || packed_kernarg == NULL || out_completion == NULL ||
      completion_size < sizeof(*out_completion) ||
      kernarg_size != kernel->code_info.kernarg_segment_size) {
    if (out_completion != NULL &&
        completion_size >= sizeof(out_completion->struct_size)) {
      out_completion->struct_size = (uint32_t)sizeof(*out_completion);
    }
    return fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "invalid managed launch resources or kernarg size");
  }
  status = validate_launch_options(options);
  if (status != SAGR_STATUS_SUCCESS) {
    return fail(out_error, error_size, status, 0,
                "invalid managed launch geometry");
  }
  session = kernel->session;
  status = ensure_launch_resources(session, out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  queue_operation(session, &operation);
  status = sagr_generic_kernarg_copy_from_host(
      kernel->kernarg, 0U, packed_kernarg, kernarg_size, &operation, out_error,
      error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  (void)sagr_generic_submit_options_init(&submit,
                                         (uint32_t)sizeof(submit));
  submit.kernarg_offset = 0U;
  submit.kernarg_size = kernarg_size;
  submit.grid_x = options->grid_x;
  submit.grid_y = options->grid_y;
  submit.grid_z = options->grid_z;
  submit.workgroup_x = options->workgroup_x;
  submit.workgroup_y = options->workgroup_y;
  submit.workgroup_z = options->workgroup_z;
  submit.num_warps = options->num_warps;
  submit.num_ctas = options->num_ctas;
  submit.shared_memory_bytes = options->shared_memory_bytes;
  submit.wavefront_size = options->wavefront_size;
  submit.launch_flags = options->launch_flags;
  memset(&ticket, 0, sizeof(ticket));
  status = sagr_queue_submit_generic_dispatch(
      session->queue, kernel->mapping, kernel->kernarg, session->signal, &submit,
      &operation, &ticket, (uint32_t)sizeof(ticket), out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  memset(out_completion, 0, completion_size);
  run_operation(session, &operation);
  status = sagr_queue_wait_generic_dispatch(
      session->queue, &ticket, &operation, out_completion, completion_size,
      out_error, error_size);
  if (status == SAGR_STATUS_SUCCESS &&
      out_completion->status == SAGR_STATUS_SUCCESS) {
    succeed(out_error, error_size,
            "managed kernel completed and resources remain reusable");
    return SAGR_STATUS_SUCCESS;
  }
  return status != SAGR_STATUS_SUCCESS ? status : out_completion->status;
}

sagr_status_t sagr_managed_kernel_unload(
    sagr_managed_kernel_t *opaque_kernel, sagr_error_info_t *out_error,
    uint32_t error_size) {
  struct sagr_managed_kernel *kernel;
  struct sagr_managed_kernel **link;
  sagr_queue_operation_options_t operation;
  sagr_status_t status = prepare_error(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (opaque_kernel == NULL) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "managed kernel pointer is null");
  }
  if (*opaque_kernel == NULL) {
    return SAGR_STATUS_SUCCESS;
  }
  kernel = (struct sagr_managed_kernel *)*opaque_kernel;
  if (!kernel_valid(kernel)) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, 0,
                "invalid managed kernel handle");
  }
  queue_operation(kernel->session, &operation);
  status = sagr_generic_unmap_object(&kernel->mapping, &operation, out_error,
                                     error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  link = &kernel->session->kernels;
  while (*link != NULL && *link != kernel) {
    link = &(*link)->next;
  }
  if (*link != kernel) {
    return fail(out_error, error_size, SAGR_STATUS_INTERNAL_ERROR, 0,
                "managed kernel ownership list is inconsistent");
  }
  *link = kernel->next;
  kernel->magic = 0U;
  *opaque_kernel = NULL;
  free(kernel);
  succeed(out_error, error_size, "managed kernel unmapped");
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_managed_session_close(
    sagr_managed_session_t *opaque_session, sagr_error_info_t *out_error,
    uint32_t error_size) {
  struct sagr_managed_session *session;
  sagr_status_t first_status = SAGR_STATUS_SUCCESS;
  sagr_error_info_t ignored_error;
  sagr_queue_operation_options_t queue_op;
  sagr_memory_operation_options_t memory_op;
  sagr_signal_operation_options_t signal_op;
  int exit_code = 0;
  sagr_status_t status = prepare_error(out_error, error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (opaque_session == NULL) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT, 0,
                "managed session pointer is null");
  }
  if (*opaque_session == NULL) {
    return SAGR_STATUS_SUCCESS;
  }
  session = (struct sagr_managed_session *)*opaque_session;
  if (!session_valid(session)) {
    return fail(out_error, error_size, SAGR_STATUS_INVALID_HANDLE, 0,
                "invalid managed session handle");
  }
  memset(&ignored_error, 0, sizeof(ignored_error));
  queue_operation(session, &queue_op);
  memory_operation(session, &memory_op);
  signal_operation(session, &signal_op);
  while (session->kernels != NULL) {
    struct sagr_managed_kernel *kernel = session->kernels;
    status = sagr_generic_unmap_object(
        &kernel->mapping, &queue_op, &ignored_error,
        (uint32_t)sizeof(ignored_error));
    if (first_status == SAGR_STATUS_SUCCESS && status != SAGR_STATUS_SUCCESS) {
      first_status = status;
    }
    session->kernels = kernel->next;
    kernel->magic = 0U;
    free(kernel);
  }
  while (session->buffers != NULL) {
    struct sagr_managed_buffer *buffer = session->buffers;
    status = sagr_memory_free(&buffer->memory, &memory_op, &ignored_error,
                              (uint32_t)sizeof(ignored_error));
    if (first_status == SAGR_STATUS_SUCCESS && status != SAGR_STATUS_SUCCESS) {
      first_status = status;
    }
    session->buffers = buffer->next;
    buffer->magic = 0U;
    free(buffer);
  }
  if (session->signal != NULL) {
    status = sagr_signal_destroy(&session->signal, &signal_op, &ignored_error,
                                 (uint32_t)sizeof(ignored_error));
    if (first_status == SAGR_STATUS_SUCCESS && status != SAGR_STATUS_SUCCESS) {
      first_status = status;
    }
  }
  if (session->queue != NULL) {
    status = sagr_queue_destroy(&session->queue, &queue_op, &ignored_error,
                                (uint32_t)sizeof(ignored_error));
    if (first_status == SAGR_STATUS_SUCCESS && status != SAGR_STATUS_SUCCESS) {
      first_status = status;
    }
  }
  status = sagr_instance_close(&session->instance);
  if (first_status == SAGR_STATUS_SUCCESS && status != SAGR_STATUS_SUCCESS) {
    first_status = status;
  }
  if (!session->external_endpoint && session->child_pid > 0 &&
      !wait_for_pid(session->child_pid, UINT64_C(2000), &exit_code)) {
    /* The connection has already been closed and owner-bound resources have
     * been released.  A private daemon that keeps listening is still owned by
     * this session, so bounded process-group termination is normal cleanup. */
    terminate_process_group(session->child_pid);
  } else if (!session->external_endpoint && session->child_pid > 0 &&
             exit_code != 0 && first_status == SAGR_STATUS_SUCCESS) {
    first_status = SAGR_STATUS_INTERNAL_ERROR;
  }
  sagr_smi_registry_release(&session->smi_lease);
  session->magic = 0U;
  session->child_pid = -1;
  session->child_pgid = -1;
  *opaque_session = NULL;
  free(session);
  if (first_status != SAGR_STATUS_SUCCESS) {
    return fail(out_error, error_size, first_status, 0,
                "managed session closed after a cleanup failure");
  }
  succeed(out_error, error_size, "managed simulator session closed cleanly");
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_managed_session_discard_inherited(
    sagr_managed_session_t *opaque_session) {
  struct sagr_managed_session *session;
  if (opaque_session == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (*opaque_session == NULL) {
    return SAGR_STATUS_SUCCESS;
  }
  session = (struct sagr_managed_session *)*opaque_session;
  if (session->magic != SAGR_MANAGED_SESSION_MAGIC ||
      session->owner_pid == getpid()) {
    return session->magic != SAGR_MANAGED_SESSION_MAGIC
               ? SAGR_STATUS_INVALID_HANDLE
               : SAGR_STATUS_INVALID_ARGUMENT;
  }
  /* sagr_instance_close only destroys this process's copied transport and
   * local handle wrappers. It deliberately sends no synthesized remote
   * teardown. The SMI atfork handler already invalidated the child lease. */
  (void)sagr_instance_close(&session->instance);
  while (session->kernels != NULL) {
    struct sagr_managed_kernel *kernel = session->kernels;
    session->kernels = kernel->next;
    kernel->magic = 0U;
    free(kernel);
  }
  while (session->buffers != NULL) {
    struct sagr_managed_buffer *buffer = session->buffers;
    session->buffers = buffer->next;
    buffer->magic = 0U;
    free(buffer);
  }
  session->queue = NULL;
  session->signal = NULL;
  session->magic = 0U;
  session->child_pid = -1;
  session->child_pgid = -1;
  *opaque_session = NULL;
  free(session);
  return SAGR_STATUS_SUCCESS;
}
