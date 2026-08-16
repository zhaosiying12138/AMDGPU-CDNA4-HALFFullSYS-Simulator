/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include "sha256_internal.h"
#include "smi_registry_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

static int read_exact(const char *path, uint8_t bytes[SAGR_SMI_RECORD_BYTES]) {
  size_t offset = 0U;
  int descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return 0;
  }
  while (offset < SAGR_SMI_RECORD_BYTES) {
    const ssize_t count = read(descriptor, bytes + offset,
                               SAGR_SMI_RECORD_BYTES - offset);
    if (count <= 0) {
      (void)close(descriptor);
      return 0;
    }
    offset += (size_t)count;
  }
  (void)close(descriptor);
  return 1;
}

static int lock_is_held(const char *path) {
  int held = 0;
  int descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return 0;
  }
  if (flock(descriptor, LOCK_SH | LOCK_NB) != 0 &&
      (errno == EWOULDBLOCK || errno == EAGAIN)) {
    held = 1;
  } else {
    (void)flock(descriptor, LOCK_UN);
  }
  (void)close(descriptor);
  return held;
}

int main(void) {
  char template_path[] = "/tmp/sagr-smi-registry-test.XXXXXX";
  char endpoint_path[108];
  char record_path[PATH_MAX];
  char released_path[PATH_MAX];
  uint8_t digest[32];
  uint8_t record[SAGR_SMI_RECORD_BYTES];
  sagr_smi_registry_identity_t identity;
  sagr_smi_registry_lease_t leases[SAGR_SMI_DEVICE_COUNT];
  sagr_smi_registry_lease_t extra;
  sagr_smi_registry_lease_t second;
  struct sockaddr_un address;
  int endpoint_descriptor = -1;
  int child_ready[2] = {-1, -1};
  int child_release[2] = {-1, -1};
  pid_t child = -1;
  int child_status = 0;
  unsigned index;
  if (mkdtemp(template_path) == NULL || chmod(template_path, 0700) != 0 ||
      snprintf(endpoint_path, sizeof(endpoint_path), "%s/bridge.sock",
               template_path) <= 0) {
    return 1;
  }
  endpoint_descriptor = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  if (endpoint_descriptor < 0 ||
      snprintf(address.sun_path, sizeof(address.sun_path), "%s",
               endpoint_path) <= 0 ||
      bind(endpoint_descriptor, (const struct sockaddr *)&address,
           sizeof(address)) != 0 ||
      chmod(endpoint_path, 0600) != 0) {
    return 1;
  }
  memset(&identity, 0, sizeof(identity));
  identity.owner_pid = getpid();
  identity.daemon_pid = getpid();
  identity.epoch = 7U;
  identity.connection_id = 11U;
  identity.world_size = 1U;
  memset(identity.job_uuid, 0x11, sizeof(identity.job_uuid));
  memset(identity.daemon_uuid, 0x22, sizeof(identity.daemon_uuid));
  identity.endpoint = endpoint_path;
  for (index = 0U; index < SAGR_SMI_DEVICE_COUNT; ++index) {
    sagr_smi_registry_lease_init(&leases[index]);
    if (!sagr_smi_registry_claim(template_path, &identity, &leases[index]) ||
        leases[index].slot != index || !lock_is_held(leases[index].path)) {
      return 2;
    }
  }
  sagr_smi_registry_lease_init(&extra);
  if (sagr_smi_registry_claim(template_path, &identity, &extra)) {
    return 3;
  }
  if (!read_exact(leases[15].path, record) ||
      memcmp(record, "SAGRSMI1", 8U) != 0 || record[19] != 15U) {
    return 4;
  }
  sagr_sha256(record, SAGR_SMI_RECORD_PAYLOAD_BYTES, digest);
  if (memcmp(digest, record + SAGR_SMI_RECORD_PAYLOAD_BYTES, 32U) != 0) {
    return 5;
  }
  if (snprintf(released_path, sizeof(released_path), "%s", leases[15].path) <=
      0) {
    return 6;
  }
  sagr_smi_registry_release(&leases[0]);
  sagr_smi_registry_release(&leases[1]);
  identity.exact_topology = 1;
  identity.rank = 15U;
  identity.world_size = 16U;
  if (!sagr_smi_registry_claim(template_path, &identity, &extra) ||
      extra.slot != 0U) {
    return 7;
  }
  sagr_smi_registry_lease_init(&second);
  identity.rank = 0U;
  if (!sagr_smi_registry_claim(template_path, &identity, &second) ||
      second.slot != 1U) {
    return 8;
  }
  if (pipe2(child_ready, O_CLOEXEC) != 0 ||
      pipe2(child_release, O_CLOEXEC) != 0) {
    return 9;
  }
  child = fork();
  if (child < 0) {
    return 9;
  }
  if (child == 0) {
    char token = 'R';
    (void)close(child_ready[0]);
    (void)close(child_release[1]);
    if (write(child_ready[1], &token, 1U) != 1 ||
        read(child_release[0], &token, 1U) != 1) {
      _exit(1);
    }
    _exit(0);
  }
  (void)close(child_ready[1]);
  (void)close(child_release[0]);
  {
    char token;
    if (read(child_ready[0], &token, 1U) != 1) {
      return 9;
    }
  }
  sagr_smi_registry_release(&leases[2]);
  identity.rank = 3U;
  if (!sagr_smi_registry_claim(template_path, &identity, &leases[0]) ||
      leases[0].slot != 2U) {
    return 9;
  }
  sagr_smi_registry_release(&leases[0]);
  {
    const char token = 'X';
    if (write(child_release[1], &token, 1U) != 1) {
      return 9;
    }
  }
  (void)close(child_ready[0]);
  (void)close(child_release[1]);
  if (waitpid(child, &child_status, 0) != child ||
      !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    return 9;
  }
  sagr_smi_registry_release(&extra);
  sagr_smi_registry_release(&second);
  for (index = 3U; index < SAGR_SMI_DEVICE_COUNT; ++index) {
    sagr_smi_registry_release(&leases[index]);
  }
  if (lock_is_held(released_path)) {
    return 9;
  }
  (void)close(endpoint_descriptor);
  (void)unlink(endpoint_path);
  for (index = 0U; index < SAGR_SMI_DEVICE_COUNT; ++index) {
    const int count = snprintf(record_path, sizeof(record_path),
                               "%s/device-%02u.bin", template_path, index);
    if (count <= 0 || (size_t)count >= sizeof(record_path) ||
        unlink(record_path) != 0) {
      return 10;
    }
  }
  (void)rmdir(template_path);
  return 0;
}
