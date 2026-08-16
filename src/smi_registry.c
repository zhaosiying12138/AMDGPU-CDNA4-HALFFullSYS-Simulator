/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include "smi_registry_internal.h"

#include "sha256_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const uint8_t record_magic[8] = {'S', 'A', 'G', 'R', 'S', 'M', 'I', '1'};
static pthread_mutex_t lease_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t atfork_once = PTHREAD_ONCE_INIT;
static sagr_smi_registry_lease_t *active_leases[SAGR_SMI_DEVICE_COUNT];
static int atfork_registration_status = -1;

static void atfork_prepare(void) { (void)pthread_mutex_lock(&lease_mutex); }

static void atfork_parent(void) { (void)pthread_mutex_unlock(&lease_mutex); }

static void atfork_child(void) {
  uint32_t index;
  for (index = 0U; index < SAGR_SMI_DEVICE_COUNT; ++index) {
    sagr_smi_registry_lease_t *lease = active_leases[index];
    if (lease != NULL) {
      if (lease->fd >= 0) {
        (void)close(lease->fd);
      }
      lease->fd = -1;
      lease->slot = UINT32_MAX;
      lease->path[0] = '\0';
      active_leases[index] = NULL;
    }
  }
  (void)pthread_mutex_unlock(&lease_mutex);
}

static void register_atfork(void) {
  atfork_registration_status =
      pthread_atfork(atfork_prepare, atfork_parent, atfork_child);
}

static int ensure_atfork(void) {
  return pthread_once(&atfork_once, register_atfork) == 0 &&
         atfork_registration_status == 0;
}

static int track_lease_locked(sagr_smi_registry_lease_t *lease) {
  uint32_t index;
  for (index = 0U; index < SAGR_SMI_DEVICE_COUNT; ++index) {
    if (active_leases[index] == NULL) {
      active_leases[index] = lease;
      return 1;
    }
  }
  return 0;
}

static void untrack_lease_locked(sagr_smi_registry_lease_t *lease) {
  uint32_t index;
  for (index = 0U; index < SAGR_SMI_DEVICE_COUNT; ++index) {
    if (active_leases[index] == lease) {
      active_leases[index] = NULL;
      break;
    }
  }
}

static void store_be32(uint8_t *bytes, uint32_t value) {
  bytes[0] = (uint8_t)(value >> 24U);
  bytes[1] = (uint8_t)(value >> 16U);
  bytes[2] = (uint8_t)(value >> 8U);
  bytes[3] = (uint8_t)value;
}

static void store_be64(uint8_t *bytes, uint64_t value) {
  unsigned index;
  for (index = 0U; index < 8U; ++index) {
    bytes[index] = (uint8_t)(value >> (56U - index * 8U));
  }
}

static int bytes_nonzero(const uint8_t *bytes, size_t size) {
  size_t index;
  for (index = 0U; index < size; ++index) {
    if (bytes[index] != 0U) {
      return 1;
    }
  }
  return 0;
}

static int private_directory(const char *path) {
  struct stat metadata;
  if (mkdir(path, S_IRWXU) != 0 && errno != EEXIST) {
    return 0;
  }
  return lstat(path, &metadata) == 0 && S_ISDIR(metadata.st_mode) &&
         metadata.st_uid == getuid() &&
         (metadata.st_mode & (S_IRWXG | S_IRWXO)) == 0;
}

static int process_start_time(pid_t pid, uint64_t *result) {
  char path[64];
  char buffer[4096];
  char *cursor;
  char *end;
  ssize_t count;
  int descriptor;
  unsigned field;
  int path_size;
  if (pid <= 0 || result == NULL) {
    return 0;
  }
  path_size = snprintf(path, sizeof(path), "/proc/%ld/stat", (long)pid);
  if (path_size < 0 || (size_t)path_size >= sizeof(path)) {
    return 0;
  }
  descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return 0;
  }
  count = read(descriptor, buffer, sizeof(buffer) - 1U);
  (void)close(descriptor);
  if (count <= 0 || (size_t)count >= sizeof(buffer)) {
    return 0;
  }
  buffer[count] = '\0';
  cursor = strrchr(buffer, ')');
  if (cursor == NULL) {
    return 0;
  }
  cursor += 1;
  for (field = 3U; field <= 22U; ++field) {
    unsigned long long value;
    while (*cursor == ' ') {
      ++cursor;
    }
    if (*cursor == '\0') {
      return 0;
    }
    end = cursor;
    while (*end != '\0' && *end != ' ') {
      ++end;
    }
    if (field == 22U) {
      char saved = *end;
      errno = 0;
      *end = '\0';
      value = strtoull(cursor, NULL, 10);
      *end = saved;
      if (errno != 0 || value == 0U) {
        return 0;
      }
      *result = (uint64_t)value;
      return 1;
    }
    cursor = end;
  }
  return 0;
}

static uint64_t monotonic_nanoseconds(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0U;
  }
  return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
         (uint64_t)value.tv_nsec;
}

static int write_all(int descriptor, const uint8_t *bytes, size_t size) {
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

static int encode_record(uint32_t slot,
                         const sagr_smi_registry_identity_t *identity,
                         uint8_t record[SAGR_SMI_RECORD_BYTES]) {
  struct stat executable;
  struct stat endpoint_metadata;
  char executable_path[64];
  int executable_path_size;
  uint64_t owner_start;
  uint64_t daemon_start;
  size_t endpoint_size;
  if (identity == NULL || identity->endpoint == NULL) {
    return 0;
  }
  endpoint_size = strlen(identity->endpoint);
  executable_path_size =
      snprintf(executable_path, sizeof(executable_path), "/proc/%ld/exe",
               (long)identity->daemon_pid);
  if (slot >= SAGR_SMI_DEVICE_COUNT || identity->owner_pid <= 0 ||
      identity->daemon_pid <= 0 || identity->epoch == 0U ||
      identity->connection_id == 0U || identity->world_size == 0U ||
      identity->world_size > SAGR_SMI_DEVICE_COUNT ||
      identity->rank >= identity->world_size ||
      identity->endpoint[0] != '/' ||
      endpoint_size == 0U || endpoint_size >= SAGR_SMI_ENDPOINT_BYTES ||
      !bytes_nonzero(identity->job_uuid, 16U) ||
      !bytes_nonzero(identity->daemon_uuid, 16U) ||
      !process_start_time(identity->owner_pid, &owner_start) ||
      !process_start_time(identity->daemon_pid, &daemon_start) ||
      executable_path_size < 0 ||
      (size_t)executable_path_size >= sizeof(executable_path) ||
      stat(executable_path, &executable) != 0 ||
      !S_ISREG(executable.st_mode) ||
      lstat(identity->endpoint, &endpoint_metadata) != 0 ||
      !S_ISSOCK(endpoint_metadata.st_mode) ||
      endpoint_metadata.st_uid != getuid() ||
      (endpoint_metadata.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
    return 0;
  }
  memset(record, 0, SAGR_SMI_RECORD_BYTES);
  memcpy(record, record_magic, sizeof(record_magic));
  store_be32(record + 8U, UINT32_C(1));
  store_be32(record + 12U, SAGR_SMI_RECORD_BYTES);
  store_be32(record + 16U, slot);
  store_be32(record + 20U, identity->exact_topology != 0 ? UINT32_C(1) : 0U);
  store_be32(record + 24U, (uint32_t)identity->owner_pid);
  store_be32(record + 28U, (uint32_t)identity->daemon_pid);
  store_be64(record + 32U, owner_start);
  store_be64(record + 40U, daemon_start);
  store_be64(record + 48U, identity->epoch);
  store_be64(record + 56U, identity->connection_id);
  store_be32(record + 64U, identity->rank);
  store_be32(record + 68U, identity->world_size);
  memcpy(record + 72U, identity->job_uuid, 16U);
  memcpy(record + 88U, identity->daemon_uuid, 16U);
  store_be64(record + 104U, (uint64_t)executable.st_dev);
  store_be64(record + 112U, (uint64_t)executable.st_ino);
  store_be64(record + 120U, monotonic_nanoseconds());
  store_be32(record + 128U, (uint32_t)endpoint_size);
  memcpy(record + 136U, identity->endpoint, endpoint_size);
  sagr_sha256(record, SAGR_SMI_RECORD_PAYLOAD_BYTES,
              record + SAGR_SMI_RECORD_PAYLOAD_BYTES);
  return 1;
}

static int acquire_slot(const char *directory, uint32_t slot,
                        const sagr_smi_registry_identity_t *identity,
                        sagr_smi_registry_lease_t *lease) {
  uint8_t record[SAGR_SMI_RECORD_BYTES];
  struct stat metadata;
  char path[PATH_MAX];
  int descriptor;
  const int flags = O_RDWR | O_CREAT | O_CLOEXEC
#ifdef O_NOFOLLOW
                    | O_NOFOLLOW
#endif
      ;
  {
    const int count =
        snprintf(path, sizeof(path), "%s/device-%02u.bin", directory, slot);
    if (count < 0 || (size_t)count >= sizeof(path) ||
        !encode_record(slot, identity, record)) {
      return 0;
    }
  }
  descriptor = open(path, flags, S_IRUSR | S_IWUSR);
  if (descriptor < 0) {
    return 0;
  }
  if (fstat(descriptor, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
      metadata.st_uid != getuid() || metadata.st_nlink != 1 ||
      fchmod(descriptor, S_IRUSR | S_IWUSR) != 0 ||
      flock(descriptor, LOCK_EX | LOCK_NB) != 0) {
    (void)close(descriptor);
    return 0;
  }
  {
    const int lease_path_size =
        snprintf(lease->path, sizeof(lease->path), "%s", path);
    if (ftruncate(descriptor, 0) != 0 || lseek(descriptor, 0, SEEK_SET) < 0 ||
        !write_all(descriptor, record, sizeof(record)) ||
        fsync(descriptor) != 0 || lease_path_size < 0 ||
        (size_t)lease_path_size >= sizeof(lease->path)) {
      (void)flock(descriptor, LOCK_UN);
      (void)close(descriptor);
      sagr_smi_registry_lease_init(lease);
      return 0;
    }
  }
  lease->fd = descriptor;
  lease->slot = slot;
  return 1;
}

void sagr_smi_registry_lease_init(sagr_smi_registry_lease_t *lease) {
  if (lease != NULL) {
    memset(lease, 0, sizeof(*lease));
    lease->fd = -1;
    lease->slot = UINT32_MAX;
  }
}

int sagr_smi_registry_default_directory(char path[PATH_MAX]) {
  const int count = snprintf(path, PATH_MAX, "/tmp/amdgpu-sim-smi-%lu",
                             (unsigned long)getuid());
  return count > 0 && count < PATH_MAX;
}

int sagr_smi_registry_claim(const char *directory,
                            const sagr_smi_registry_identity_t *identity,
                            sagr_smi_registry_lease_t *lease) {
  char default_directory[PATH_MAX];
  uint32_t slot;
  if (identity == NULL || lease == NULL || lease->fd >= 0 ||
      !ensure_atfork()) {
    return 0;
  }
  if (directory == NULL) {
    if (!sagr_smi_registry_default_directory(default_directory)) {
      return 0;
    }
    directory = default_directory;
  }
  if (directory[0] != '/' || !private_directory(directory)) {
    return 0;
  }
  if (pthread_mutex_lock(&lease_mutex) != 0) {
    return 0;
  }
  /* A topology rank is not a machine-wide device ordinal.  Always allocate the
   * first free logical device so independent TP jobs can coexist. */
  for (slot = 0U; slot < SAGR_SMI_DEVICE_COUNT; ++slot) {
    if (acquire_slot(directory, slot, identity, lease)) {
      if (track_lease_locked(lease)) {
        (void)pthread_mutex_unlock(&lease_mutex);
        return 1;
      }
      (void)flock(lease->fd, LOCK_UN);
      (void)close(lease->fd);
      sagr_smi_registry_lease_init(lease);
      (void)pthread_mutex_unlock(&lease_mutex);
      return 0;
    }
  }
  (void)pthread_mutex_unlock(&lease_mutex);
  return 0;
}

void sagr_smi_registry_release(sagr_smi_registry_lease_t *lease) {
  if (lease == NULL) {
    return;
  }
  if (pthread_mutex_lock(&lease_mutex) != 0) {
    return;
  }
  if (lease->fd >= 0) {
    untrack_lease_locked(lease);
    (void)flock(lease->fd, LOCK_UN);
    (void)close(lease->fd);
  }
  sagr_smi_registry_lease_init(lease);
  (void)pthread_mutex_unlock(&lease_mutex);
}
