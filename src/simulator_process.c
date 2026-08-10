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
  const char *gem5;
  const char *config;
  const char *repo;
  const char *endpoint;
  memset(simulator, 0, sizeof(*simulator));
  prefix = getenv("SAGR_OPENCL_PREFIX");
  gem5 = getenv("SAGR_OPENCL_GEM5");
  config = getenv("SAGR_OPENCL_GEM5_CONFIG");
  repo = getenv("SAGR_OPENCL_REPO_ROOT");
  endpoint = getenv("SAGR_GENERIC_BRIDGE_ENDPOINT");
  if (!copy_path(simulator->prefix,
                 prefix != NULL && prefix[0] != '\0'
                     ? prefix
                     : SAGR_OPENCL_DEFAULT_PREFIX) ||
      !copy_path(simulator->gem5_path,
                 gem5 != NULL && gem5[0] != '\0' ? gem5
                                                  : SAGR_OPENCL_DEFAULT_GEM5) ||
      !copy_path(simulator->gem5_config_path,
                 config != NULL && config[0] != '\0'
                     ? config
                     : SAGR_OPENCL_DEFAULT_GEM5_CONFIG) ||
      !copy_path(simulator->repo_root,
                 repo != NULL && repo[0] != '\0'
                     ? repo
                     : SAGR_OPENCL_DEFAULT_REPO_ROOT) ||
      !join_path(simulator->clang_path, simulator->prefix, "bin/clang")) {
    return;
  }
  if (endpoint != NULL && endpoint[0] != '\0') {
    if (!copy_path(simulator->endpoint, endpoint)) {
      return;
    }
    simulator->external_endpoint = 1;
  }
  simulator->paths_ready = 1;
}

static int regular_file(const char *path, int executable) {
  struct stat state;
  return stat(path, &state) == 0 && S_ISREG(state.st_mode) &&
         (!executable || access(path, X_OK) == 0);
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

static int spawn_gem5(struct sagr_cl_simulator *simulator) {
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  char epoch[32];
  char path_environment[PATH_MAX + 32];
  char home_environment[PATH_MAX + 16];
  char temp_environment[PATH_MAX + 16];
  char cache_environment[PATH_MAX + 32];
  char *environment[8];
  char *arguments[28];
  int index = 0;
  int count;
  if (!regular_file(simulator->gem5_path, 1) ||
      !regular_file(simulator->gem5_config_path, 0) ||
      access(simulator->repo_root, X_OK) != 0) {
    return 0;
  }
  count = snprintf(epoch, sizeof(epoch), "%llu",
                   (unsigned long long)simulator->epoch);
  if (count < 0 || (size_t)count >= sizeof(epoch) ||
      snprintf(path_environment, sizeof(path_environment),
               "PATH=/usr/bin:/bin") < 0 ||
      snprintf(home_environment, sizeof(home_environment), "HOME=%s",
               simulator->run_dir) < 0 ||
      snprintf(temp_environment, sizeof(temp_environment), "TMPDIR=%s",
               simulator->run_dir) < 0 ||
      snprintf(cache_environment, sizeof(cache_environment),
               "XDG_CACHE_HOME=%s/cache", simulator->run_dir) < 0) {
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

  arguments[index++] = simulator->gem5_path;
  arguments[index++] = (char *)"--listener-mode=on";
  arguments[index++] = (char *)"--outdir";
  arguments[index++] = simulator->output_dir;
  arguments[index++] = simulator->gem5_config_path;
  arguments[index++] = (char *)"--endpoint";
  arguments[index++] = simulator->endpoint;
  arguments[index++] = (char *)"--dispatch-trace-path";
  arguments[index++] = simulator->trace_path;
  arguments[index++] = (char *)"--epoch";
  arguments[index++] = epoch;
  arguments[index++] = (char *)"--job-uuid";
  arguments[index++] = simulator->job_uuid_text;
  arguments[index++] = (char *)"--rank";
  arguments[index++] = (char *)"0";
  arguments[index++] = (char *)"--world-size";
  arguments[index++] = (char *)"1";
  arguments[index++] = (char *)"--startup-timeout-ms";
  arguments[index++] = (char *)"15000";
  arguments[index++] = (char *)"--handshake-timeout-ms";
  arguments[index++] = (char *)"10000";
  arguments[index++] = (char *)"--run-timeout-ms";
  arguments[index++] = (char *)"120000";
  arguments[index] = NULL;

  if (!configure_spawn(&actions, &attributes, simulator->repo_root,
                       simulator->log_path)) {
    return 0;
  }
  if (posix_spawn(&simulator->child_pid, simulator->gem5_path, &actions,
                  &attributes, arguments, environment) != 0) {
    simulator->child_pid = -1;
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  (void)posix_spawnattr_destroy(&attributes);
  if (simulator->child_pid <= 0) {
    return 0;
  }
  simulator->child_pgid = simulator->child_pid;
  return 1;
}

static int wait_for_endpoint(struct sagr_cl_simulator *simulator) {
  const uint64_t start = monotonic_milliseconds();
  struct timespec delay = {0, 50000000L};
  for (;;) {
    struct stat state;
    int status = 0;
    const pid_t exited = waitpid(simulator->child_pid, &status, WNOHANG);
    if (exited == simulator->child_pid || (exited < 0 && errno != EINTR)) {
      simulator->child_pid = -1;
      simulator->child_pgid = -1;
      return 0;
    }
    if (lstat(simulator->endpoint, &state) == 0 && S_ISSOCK(state.st_mode) &&
        state.st_uid == getuid() && (state.st_mode & (S_IRWXG | S_IRWXO)) == 0) {
      return 1;
    }
    if (monotonic_milliseconds() - start >= SAGR_CL_STARTUP_TIMEOUT_MS) {
      return 0;
    }
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {
    }
    delay.tv_sec = 0;
    delay.tv_nsec = 50000000L;
  }
}

static cl_int open_instance(struct sagr_cl_simulator *simulator) {
  const uint64_t capabilities =
      SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK |
      SAGR_CAPABILITY_MEMORY_MASK | SAGR_CAPABILITY_SIGNAL_MASK |
      SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK |
      SAGR_CAPABILITY_GENERIC_DISPATCH_MASK |
      SAGR_CAPABILITY_GENERIC_EXECUTION_MASK;
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  sagr_status_t status;
  if (sagr_instance_open_options_init(&options, (uint32_t)sizeof(options)) !=
      SAGR_STATUS_SUCCESS) {
    return CL_OUT_OF_RESOURCES;
  }
  options.open_timeout_ns = SAGR_CL_STARTUP_TIMEOUT_MS * UINT64_C(1000000);
  options.offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |= capabilities;
  options.required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |= capabilities;
  if (!simulator->external_endpoint) {
    memcpy(options.expected_job_uuid, simulator->job_uuid,
           sizeof(options.expected_job_uuid));
    options.expected_epoch = simulator->epoch;
    options.expected_rank = 0U;
    options.expected_world_size = 1U;
  }
  memset(&error, 0, sizeof(error));
  status = sagr_instance_open(simulator->endpoint, &options,
                              &simulator->instance, &error,
                              (uint32_t)sizeof(error));
  return sagr_cl_status_to_error(status);
}

cl_int sagr_cl_simulator_ensure(struct sagr_cl_simulator *simulator) {
  char template_path[PATH_MAX];
  char cache_path[PATH_MAX];
  cl_int result;
  int count;
  if (simulator->instance != NULL) {
    return CL_SUCCESS;
  }
  if (!simulator->paths_ready) {
    return CL_OUT_OF_RESOURCES;
  }
  if (simulator->external_endpoint) {
    result = open_instance(simulator);
    if (result == CL_SUCCESS) {
      simulator->started = 1;
    }
    return result;
  }
  count = snprintf(template_path, sizeof(template_path),
                   "/tmp/self-amdgpu-opencl-run.%lu.XXXXXX",
                   (unsigned long)getuid());
  if (count < 0 || (size_t)count >= sizeof(template_path) ||
      mkdtemp(template_path) == NULL || chmod(template_path, S_IRWXU) != 0 ||
      !copy_path(simulator->run_dir, template_path) ||
      !join_path(simulator->endpoint, simulator->run_dir, "bridge.sock") ||
      !join_path(simulator->trace_path, simulator->run_dir,
                 "dispatch-trace.jsonl") ||
      !join_path(simulator->output_dir, simulator->run_dir, "m5out") ||
      !join_path(simulator->log_path, simulator->run_dir, "gem5.log") ||
      !join_path(cache_path, simulator->run_dir, "cache") ||
      mkdir(simulator->output_dir, S_IRWXU) != 0 ||
      mkdir(cache_path, S_IRWXU) != 0) {
    return CL_OUT_OF_RESOURCES;
  }
  simulator->epoch = make_epoch();
  sagr_cl_make_job_uuid(simulator->epoch, (uint64_t)(uint32_t)getpid(),
                        simulator->job_uuid, simulator->job_uuid_text);
  if (!spawn_gem5(simulator) || !wait_for_endpoint(simulator)) {
    terminate_process_group(simulator->child_pid);
    simulator->child_pid = -1;
    simulator->child_pgid = -1;
    return CL_OUT_OF_RESOURCES;
  }
  result = open_instance(simulator);
  if (result != CL_SUCCESS) {
    terminate_process_group(simulator->child_pid);
    simulator->child_pid = -1;
    simulator->child_pgid = -1;
    return result;
  }
  simulator->started = 1;
  return CL_SUCCESS;
}

void sagr_cl_simulator_shutdown(struct sagr_cl_simulator *simulator) {
  int exit_code = 0;
  if (simulator->instance != NULL) {
    (void)sagr_instance_close(&simulator->instance);
  }
  if (!simulator->external_endpoint && simulator->child_pid > 0) {
    if (!wait_for_pid(simulator->child_pid, SAGR_CL_STARTUP_TIMEOUT_MS,
                      &exit_code)) {
      terminate_process_group(simulator->child_pid);
    }
  }
  simulator->child_pid = -1;
  simulator->child_pgid = -1;
  simulator->started = 0;
}
