/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_SMI_REGISTRY_INTERNAL_H
#define SELF_AMDGPU_RUNTIME_SMI_REGISTRY_INTERNAL_H

#include <limits.h>
#include <stdint.h>
#include <sys/types.h>

enum {
  SAGR_SMI_DEVICE_COUNT = 16,
  SAGR_SMI_RECORD_BYTES = 320,
  SAGR_SMI_RECORD_PAYLOAD_BYTES = 288,
  SAGR_SMI_ENDPOINT_BYTES = 112
};

typedef struct sagr_smi_registry_identity {
  pid_t owner_pid;
  pid_t daemon_pid;
  uint64_t epoch;
  uint64_t connection_id;
  uint32_t rank;
  uint32_t world_size;
  uint8_t job_uuid[16];
  uint8_t daemon_uuid[16];
  const char *endpoint;
  int exact_topology;
} sagr_smi_registry_identity_t;

typedef struct sagr_smi_registry_lease {
  int fd;
  uint32_t slot;
  char path[PATH_MAX];
} sagr_smi_registry_lease_t;

void sagr_smi_registry_lease_init(sagr_smi_registry_lease_t *lease);
int sagr_smi_registry_default_directory(char path[PATH_MAX]);
int sagr_smi_registry_claim(const char *directory,
                            const sagr_smi_registry_identity_t *identity,
                            sagr_smi_registry_lease_t *lease);
void sagr_smi_registry_release(sagr_smi_registry_lease_t *lease);

#endif
