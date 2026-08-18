/* SPDX-License-Identifier: GPL-3.0-or-later */

/* The managed run directory is the only durable identity a simulator carries.
 * gem5 is spawned with a scrubbed environment (PATH, HOME, TMPDIR,
 * XDG_CACHE_HOME, LC_ALL -- nothing else), and when the process that started it
 * dies the simulator is reparented to init, so neither the environment nor the
 * process tree can say which run a given simulator belongs to. Only its run
 * directory can.
 *
 * That makes the run root operationally load-bearing: with every run sharing
 * one /tmp, concurrent runs are indistinguishable, and two real failures
 * followed -- retired dispatches summed across unrelated runs, and a cleanup
 * that killed a healthy run because it could not tell that run's simulators
 * from an abandoned run's. SAGR_MANAGED_RUN_ROOT gives each run its own root
 * so both operations become exact.
 *
 * These tests drive the real public entry point. The run directory is created
 * before gem5 is spawned and is deliberately left behind when startup fails,
 * so pointing SAGR_MANAGED_GEM5 at a program that exits immediately exercises
 * the directory placement without paying for a simulator.
 */

#include "self_amdgpu_runtime/runtime.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *const kRunPrefix = "self-amdgpu-opencl-run.";

/* Count entries under `root` whose name looks like a managed run directory. */
static int count_run_dirs(const char *root) {
  DIR *directory = opendir(root);
  struct dirent *entry;
  int found = 0;
  if (directory == NULL) {
    return -1;
  }
  while ((entry = readdir(directory)) != NULL) {
    if (strncmp(entry->d_name, kRunPrefix, strlen(kRunPrefix)) == 0) {
      found++;
    }
  }
  (void)closedir(directory);
  return found;
}

static void remove_run_dirs(const char *root) {
  DIR *directory = opendir(root);
  struct dirent *entry;
  char path[4096];
  if (directory == NULL) {
    return;
  }
  while ((entry = readdir(directory)) != NULL) {
    if (strncmp(entry->d_name, kRunPrefix, strlen(kRunPrefix)) != 0) {
      continue;
    }
    /* The run directory holds only directories the session just made. */
    if (snprintf(path, sizeof(path), "rm -rf '%s/%s'", root, entry->d_name) <
        (int)sizeof(path)) {
      (void)system(path);
    }
  }
  (void)closedir(directory);
}

/* Attempt a managed session that is guaranteed to fail during gem5 startup,
 * leaving its run directory in place. Returns the open status. */
static sagr_status_t open_doomed_session(const char *run_root) {
  sagr_managed_session_t session = NULL;
  sagr_error_info_t error;
  sagr_status_t status;

  if (run_root != NULL) {
    (void)setenv("SAGR_MANAGED_RUN_ROOT", run_root, 1);
  } else {
    (void)unsetenv("SAGR_MANAGED_RUN_ROOT");
  }
  (void)unsetenv("SAGR_OPENCL_RUN_ROOT");
  /* /bin/false publishes no endpoint, so startup fails in seconds. */
  (void)setenv("SAGR_MANAGED_GEM5", "/bin/false", 1);
  (void)unsetenv("SAGR_OPENCL_GEM5_EXTERNAL");
  (void)unsetenv("SAGR_OPENCL_ENDPOINT");
  (void)unsetenv("SAGR_OPENCL_SOCKET");

  memset(&error, 0, sizeof(error));
  status = sagr_managed_session_open(NULL, &session, NULL, 0U, &error,
                                     (uint32_t)sizeof(error));
  if (status == SAGR_STATUS_SUCCESS) {
    (void)sagr_managed_session_close(&session, NULL, 0U);
  }
  return status;
}

static int test_run_root_places_the_run_directory(void) {
  char root[] = "/tmp/sagr-run-root-test.XXXXXX";
  int found;

  if (mkdtemp(root) == NULL) {
    fprintf(stderr, "could not create a temporary run root\n");
    return 1;
  }
  if (open_doomed_session(root) == SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "a session with /bin/false as gem5 unexpectedly opened\n");
    remove_run_dirs(root);
    (void)rmdir(root);
    return 1;
  }
  found = count_run_dirs(root);
  remove_run_dirs(root);
  (void)rmdir(root);
  if (found != 1) {
    fprintf(stderr,
            "expected exactly one run directory under the requested root, "
            "found %d\n",
            found);
    return 1;
  }
  return 0;
}

/* A run root that does not exist yet must be created rather than rejected: the
 * caller is a lane script that names a per-lane directory up front. */
static int test_run_root_is_created_when_absent(void) {
  char parent[] = "/tmp/sagr-run-root-test.XXXXXX";
  char root[4096];
  int found;

  if (mkdtemp(parent) == NULL) {
    fprintf(stderr, "could not create a temporary run root parent\n");
    return 1;
  }
  if (snprintf(root, sizeof(root), "%s/not-created-yet", parent) >=
      (int)sizeof(root)) {
    (void)rmdir(parent);
    return 1;
  }
  (void)open_doomed_session(root);
  found = count_run_dirs(root);
  remove_run_dirs(root);
  (void)rmdir(root);
  (void)rmdir(parent);
  if (found != 1) {
    fprintf(stderr,
            "expected the absent run root to be created and used, found %d\n",
            found);
    return 1;
  }
  return 0;
}

/* Without the variable the historical /tmp placement must be preserved, so an
 * unconfigured caller keeps working exactly as before. */
static int test_default_root_is_tmp(void) {
  const int before = count_run_dirs("/tmp");
  int after;

  if (before < 0) {
    fprintf(stderr, "could not read /tmp\n");
    return 1;
  }
  (void)open_doomed_session(NULL);
  after = count_run_dirs("/tmp");
  if (after != before + 1) {
    fprintf(stderr,
            "expected the default root to stay /tmp: %d run directories "
            "before, %d after\n",
            before, after);
    return 1;
  }
  return 0;
}

int main(void) {
  int failures = 0;
  failures += test_run_root_places_the_run_directory();
  failures += test_run_root_is_created_when_absent();
  failures += test_default_root_is_tmp();
  return failures == 0 ? 0 : 1;
}
