/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include "managed_supervisor_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/signalfd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

static uint64_t monotonic_milliseconds(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0U;
  }
  return (uint64_t)value.tv_sec * UINT64_C(1000) +
         (uint64_t)value.tv_nsec / UINT64_C(1000000);
}

static int parse_positive_pid(const char *text, pid_t *result) {
  char *end = NULL;
  long value;
  if (text == NULL || result == NULL || text[0] == '\0') {
    return 0;
  }
  errno = 0;
  value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0L ||
      value > (long)INT32_MAX) {
    return 0;
  }
  *result = (pid_t)value;
  return 1;
}

static int parse_descriptor(const char *text, int *result) {
  char *end = NULL;
  long value;
  if (text == NULL || result == NULL || text[0] == '\0') {
    return 0;
  }
  errno = 0;
  value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 3L ||
      value > (long)INT_MAX) {
    return 0;
  }
  *result = (int)value;
  return 1;
}

static int parse_grace(const char *text, uint64_t *result) {
  char *end = NULL;
  unsigned long long value;
  if (text == NULL || result == NULL || text[0] == '\0') {
    return 0;
  }
  errno = 0;
  value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0U ||
      value > (unsigned long long)SAGR_MANAGED_SUPERVISOR_MAX_GRACE_MS) {
    return 0;
  }
  *result = (uint64_t)value;
  return 1;
}

static int write_all(int descriptor, const void *buffer, size_t size) {
  const uint8_t *bytes = (const uint8_t *)buffer;
  size_t offset = 0U;
  while (offset < size) {
    const ssize_t count = write(descriptor, bytes + offset, size - offset);
    if (count > 0) {
      offset += (size_t)count;
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      return 0;
    }
  }
  return 1;
}

static int report_startup(int descriptor, int error_number, pid_t daemon_pid) {
  sagr_managed_supervisor_report_t report;
  memset(&report, 0, sizeof(report));
  report.magic = SAGR_MANAGED_SUPERVISOR_REPORT_MAGIC;
  report.version = SAGR_MANAGED_SUPERVISOR_PROTOCOL_VERSION;
  report.error_number = error_number;
  report.daemon_pid = (int64_t)daemon_pid;
  return write_all(descriptor, &report, sizeof(report));
}

static int owner_is_dead(int pid_descriptor) {
  struct pollfd descriptor;
  int result;
  memset(&descriptor, 0, sizeof(descriptor));
  descriptor.fd = pid_descriptor;
  descriptor.events = POLLIN;
  do {
    result = poll(&descriptor, 1U, 0);
  } while (result < 0 && errno == EINTR);
  return result > 0 &&
         (descriptor.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) != 0;
}

static int wait_for_child(pid_t child, uint64_t timeout_ms, int *status) {
  const uint64_t start = monotonic_milliseconds();
  struct timespec delay = {0, 10000000L};
  for (;;) {
    const pid_t result = waitpid(child, status, WNOHANG);
    if (result == child) {
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

static int child_exit_code(int status) {
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    const int value = 128 + WTERMSIG(status);
    return value <= 255 ? value : 255;
  }
  return 125;
}

static void terminate_daemon(pid_t child, uint64_t grace_ms) {
  int status = 0;
  if (child <= 0 || waitpid(child, &status, WNOHANG) == child) {
    return;
  }
  if (kill(-child, SIGTERM) != 0 && errno == ESRCH) {
    (void)kill(child, SIGTERM);
  }
  if (!wait_for_child(child, grace_ms, &status)) {
    if (kill(-child, SIGKILL) != 0 && errno == ESRCH) {
      (void)kill(child, SIGKILL);
    }
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
  }
}

static void child_exec_failure(int descriptor, int error_number) {
  (void)write_all(descriptor, &error_number, sizeof(error_number));
  _exit(127);
}

static pid_t spawn_daemon(char *const arguments[], int report_descriptor,
                          int owner_pid_descriptor,
                          int signal_descriptor, int *error_number) {
  int exec_pipe[2] = {-1, -1};
  pid_t child;
  int child_error = 0;
  ssize_t count;
  if (pipe2(exec_pipe, O_CLOEXEC) != 0) {
    *error_number = errno;
    return -1;
  }
  child = fork();
  if (child < 0) {
    *error_number = errno;
    (void)close(exec_pipe[0]);
    (void)close(exec_pipe[1]);
    return -1;
  }
  if (child == 0) {
    struct sigaction default_action;
    sigset_t empty_mask;
    const pid_t supervisor_pid = getppid();
    (void)close(exec_pipe[0]);
    (void)close(report_descriptor);
    (void)close(owner_pid_descriptor);
    (void)close(signal_descriptor);
    if (prctl(PR_SET_PDEATHSIG, (unsigned long)SIGKILL, 0UL, 0UL, 0UL) != 0 ||
        getppid() != supervisor_pid || setpgid(0, 0) != 0) {
      child_exec_failure(exec_pipe[1], errno != 0 ? errno : ESRCH);
    }
    memset(&default_action, 0, sizeof(default_action));
    default_action.sa_handler = SIG_DFL;
    (void)sigemptyset(&default_action.sa_mask);
    if (sigaction(SIGTERM, &default_action, NULL) != 0 ||
        sigaction(SIGUSR1, &default_action, NULL) != 0 ||
        sigaction(SIGCHLD, &default_action, NULL) != 0 ||
        sigemptyset(&empty_mask) != 0 ||
        sigprocmask(SIG_SETMASK, &empty_mask, NULL) != 0) {
      child_exec_failure(exec_pipe[1], errno);
    }
    execve(arguments[0], arguments, environ);
    child_exec_failure(exec_pipe[1], errno);
  }
  (void)close(exec_pipe[1]);
  do {
    count = read(exec_pipe[0], &child_error, sizeof(child_error));
  } while (count < 0 && errno == EINTR);
  (void)close(exec_pipe[0]);
  if (count == 0) {
    return child;
  }
  *error_number = count == (ssize_t)sizeof(child_error) && child_error != 0
                      ? child_error
                      : EIO;
  while (waitpid(child, NULL, 0) < 0 && errno == EINTR) {
  }
  return -1;
}

static int supervise(pid_t owner_pid, int owner_pid_descriptor,
                     int signal_descriptor, pid_t daemon_pid,
                     uint64_t grace_ms) {
  struct pollfd descriptors[2];
  for (;;) {
    struct signalfd_siginfo signal_info;
    int status = 0;
    const pid_t exited = waitpid(daemon_pid, &status, WNOHANG);
    if (exited == daemon_pid) {
      return child_exit_code(status);
    }
    if (exited < 0 && errno != EINTR) {
      return 125;
    }
    memset(descriptors, 0, sizeof(descriptors));
    descriptors[0].fd = owner_pid_descriptor;
    descriptors[0].events = POLLIN;
    descriptors[1].fd = signal_descriptor;
    descriptors[1].events = POLLIN;
    if (poll(descriptors, 2U, -1) < 0) {
      if (errno == EINTR) {
        continue;
      }
      terminate_daemon(daemon_pid, grace_ms);
      return 125;
    }
    if ((descriptors[0].revents &
         (POLLIN | POLLHUP | POLLERR | POLLNVAL)) != 0) {
      terminate_daemon(daemon_pid, grace_ms);
      return 0;
    }
    if ((descriptors[1].revents & POLLIN) == 0) {
      continue;
    }
    if (read(signal_descriptor, &signal_info, sizeof(signal_info)) !=
        (ssize_t)sizeof(signal_info)) {
      if (errno == EINTR) {
        continue;
      }
      terminate_daemon(daemon_pid, grace_ms);
      return 125;
    }
    if (signal_info.ssi_signo == (uint32_t)SIGTERM ||
        signal_info.ssi_signo == (uint32_t)SIGINT ||
        signal_info.ssi_signo == (uint32_t)SIGHUP) {
      terminate_daemon(daemon_pid, grace_ms);
      return 0;
    }
    if (signal_info.ssi_signo == (uint32_t)SIGUSR1) {
      /* PR_SET_PDEATHSIG follows the creating thread, not the process. The
       * process pidfd is authoritative and prevents a short-lived Python
       * launcher thread from terminating a live session. */
      if (owner_is_dead(owner_pid_descriptor)) {
        terminate_daemon(daemon_pid, grace_ms);
        return 0;
      }
      continue;
    }
    if (signal_info.ssi_signo == (uint32_t)SIGCHLD) {
      continue;
    }
    if (getppid() != owner_pid && owner_is_dead(owner_pid_descriptor)) {
      terminate_daemon(daemon_pid, grace_ms);
      return 0;
    }
  }
}

int main(int argc, char **argv) {
  pid_t owner_pid = -1;
  pid_t daemon_pid = -1;
  int report_descriptor = -1;
  int owner_pid_descriptor = -1;
  int signal_descriptor = -1;
  int error_number = EINVAL;
  int separator = -1;
  uint64_t grace_ms = 0U;
  sigset_t signal_mask;
  struct sigaction default_action;
  int index;
  if (argc < 10 || strcmp(argv[1], "--owner-pid") != 0 ||
      strcmp(argv[3], "--status-fd") != 0 ||
      strcmp(argv[5], "--grace-ms") != 0 ||
      !parse_positive_pid(argv[2], &owner_pid) ||
      !parse_descriptor(argv[4], &report_descriptor) ||
      !parse_grace(argv[6], &grace_ms)) {
    return 125;
  }
  for (index = 7; index < argc; ++index) {
    if (strcmp(argv[index], "--") == 0) {
      separator = index;
      break;
    }
  }
  if (separator < 0 || separator + 1 >= argc || argv[separator + 1][0] != '/' ||
      fcntl(report_descriptor, F_GETFD) < 0) {
    (void)report_startup(report_descriptor, EINVAL, -1);
    return 125;
  }
  memset(&default_action, 0, sizeof(default_action));
  default_action.sa_handler = SIG_DFL;
  (void)sigemptyset(&default_action.sa_mask);
  if (sigaction(SIGCHLD, &default_action, NULL) != 0 ||
      sigemptyset(&signal_mask) != 0 ||
      sigaddset(&signal_mask, SIGTERM) != 0 ||
      sigaddset(&signal_mask, SIGINT) != 0 ||
      sigaddset(&signal_mask, SIGHUP) != 0 ||
      sigaddset(&signal_mask, SIGUSR1) != 0 ||
      sigaddset(&signal_mask, SIGCHLD) != 0 ||
      sigprocmask(SIG_BLOCK, &signal_mask, NULL) != 0) {
    error_number = errno;
    (void)report_startup(report_descriptor, error_number, -1);
    return 125;
  }
#if defined(SYS_pidfd_open)
  owner_pid_descriptor = (int)syscall(SYS_pidfd_open, owner_pid, 0U);
#else
  errno = ENOSYS;
  owner_pid_descriptor = -1;
#endif
  if (owner_pid_descriptor < 0 || getppid() != owner_pid ||
      prctl(PR_SET_PDEATHSIG, (unsigned long)SIGUSR1, 0UL, 0UL, 0UL) != 0 ||
      getppid() != owner_pid || owner_is_dead(owner_pid_descriptor)) {
    error_number = errno != 0 ? errno : ESRCH;
    (void)report_startup(report_descriptor, error_number, -1);
    if (owner_pid_descriptor >= 0) {
      (void)close(owner_pid_descriptor);
    }
    return 125;
  }
  signal_descriptor = signalfd(-1, &signal_mask, SFD_CLOEXEC);
  if (signal_descriptor < 0) {
    error_number = errno;
    (void)report_startup(report_descriptor, error_number, -1);
    (void)close(owner_pid_descriptor);
    return 125;
  }
  daemon_pid = spawn_daemon(&argv[separator + 1], report_descriptor,
                            owner_pid_descriptor, signal_descriptor,
                            &error_number);
  if (daemon_pid <= 0) {
    (void)report_startup(report_descriptor, error_number, -1);
    (void)close(signal_descriptor);
    (void)close(owner_pid_descriptor);
    return 125;
  }
  if (!report_startup(report_descriptor, 0, daemon_pid)) {
    terminate_daemon(daemon_pid, grace_ms);
    (void)close(signal_descriptor);
    (void)close(owner_pid_descriptor);
    return 125;
  }
  (void)close(report_descriptor);
  error_number = supervise(owner_pid, owner_pid_descriptor, signal_descriptor,
                           daemon_pid, grace_ms);
  (void)close(signal_descriptor);
  (void)close(owner_pid_descriptor);
  return error_number;
}
