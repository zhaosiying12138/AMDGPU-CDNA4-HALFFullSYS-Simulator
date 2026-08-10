/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include "opencl_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef SAGR_OPENCL_DEFAULT_PREFIX
#define SAGR_OPENCL_DEFAULT_PREFIX ""
#endif
#ifndef SAGR_OPENCL_DEFAULT_GEM5
#define SAGR_OPENCL_DEFAULT_GEM5 ""
#endif
#ifndef SAGR_OPENCL_DEFAULT_GEM5_CONFIG
#define SAGR_OPENCL_DEFAULT_GEM5_CONFIG ""
#endif
#ifndef SAGR_OPENCL_DEFAULT_REPO_ROOT
#define SAGR_OPENCL_DEFAULT_REPO_ROOT ""
#endif

static uint64_t monotonic_milliseconds(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0U;
  }
  return (uint64_t)value.tv_sec * UINT64_C(1000) +
         (uint64_t)value.tv_nsec / UINT64_C(1000000);
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

static int configure_spawn(posix_spawn_file_actions_t *actions,
                           posix_spawnattr_t *attributes,
                           const char *working_dir, const char *log_path) {
  short flags = POSIX_SPAWN_SETPGROUP;
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
  return 1;
}

int sagr_cl_spawn_and_wait(const char *path, char *const argv[],
                           char *const environment[], const char *working_dir,
                           const char *log_path, uint64_t timeout_ms,
                           int *exit_code) {
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  pid_t child = -1;
  int actions_ready = 0;
  int attributes_ready = 0;
  int result = 0;
  if (path == NULL || path[0] != '/' || argv == NULL || environment == NULL ||
      log_path == NULL || log_path[0] != '/' || timeout_ms == 0U) {
    return 0;
  }
  if (posix_spawn_file_actions_init(&actions) == 0) {
    actions_ready = 1;
  }
  if (posix_spawnattr_init(&attributes) == 0) {
    attributes_ready = 1;
  }
  if (!actions_ready || !attributes_ready) {
    goto cleanup;
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  (void)posix_spawnattr_destroy(&attributes);
  actions_ready = 0;
  attributes_ready = 0;
  if (!configure_spawn(&actions, &attributes, working_dir, log_path)) {
    goto cleanup;
  }
  actions_ready = 1;
  attributes_ready = 1;
  if (posix_spawn(&child, path, &actions, &attributes, argv, environment) !=
      0) {
    child = -1;
    goto cleanup;
  }
  if (!wait_for_pid(child, timeout_ms, exit_code)) {
    terminate_process_group(child);
    child = -1;
    goto cleanup;
  }
  child = -1;
  result = 1;

cleanup:
  if (child > 0) {
    terminate_process_group(child);
  }
  if (actions_ready) {
    (void)posix_spawn_file_actions_destroy(&actions);
  }
  if (attributes_ready) {
    (void)posix_spawnattr_destroy(&attributes);
  }
  return result;
}

void sagr_cl_simulator_init(struct sagr_cl_simulator *simulator) {
  const char *prefix;
  memset(simulator, 0, sizeof(*simulator));
  prefix = getenv("SAGR_OPENCL_PREFIX");
  if (!copy_path(simulator->prefix,
                 prefix != NULL && prefix[0] != '\0'
                     ? prefix
                     : SAGR_OPENCL_DEFAULT_PREFIX) ||
      !join_path(simulator->clang_path, simulator->prefix, "bin/clang")) {
    return;
  }
  simulator->paths_ready = 1;
}

cl_int sagr_cl_simulator_ensure(struct sagr_cl_simulator *simulator) {
  if (simulator->instance != NULL) {
    return CL_SUCCESS;
  }
  if (!simulator->paths_ready) {
    return CL_OUT_OF_RESOURCES;
  }
  if (sagr_managed_session_open(NULL, &simulator->managed_session, NULL, 0U,
                                NULL, 0U) != SAGR_STATUS_SUCCESS ||
      sagr_managed_session_get_instance(simulator->managed_session,
                                        &simulator->instance) !=
          SAGR_STATUS_SUCCESS) {
    if (simulator->managed_session != NULL) {
      (void)sagr_managed_session_close(&simulator->managed_session, NULL, 0U);
    }
    simulator->instance = NULL;
    return CL_OUT_OF_RESOURCES;
  }
  return CL_SUCCESS;
}

void sagr_cl_simulator_shutdown(struct sagr_cl_simulator *simulator) {
  simulator->instance = NULL;
  if (simulator->managed_session != NULL) {
    (void)sagr_managed_session_close(&simulator->managed_session, NULL, 0U);
  }
}
