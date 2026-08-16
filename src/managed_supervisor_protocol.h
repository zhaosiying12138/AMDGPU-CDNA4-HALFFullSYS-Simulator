/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_MANAGED_SUPERVISOR_PROTOCOL_H
#define SELF_AMDGPU_RUNTIME_MANAGED_SUPERVISOR_PROTOCOL_H

#include <stdint.h>

#define SAGR_MANAGED_SUPERVISOR_REPORT_MAGIC \
  UINT64_C(0x5341475253555031)

enum {
  SAGR_MANAGED_SUPERVISOR_PROTOCOL_VERSION = 1,
  SAGR_MANAGED_SUPERVISOR_REPORT_FD = 3,
  SAGR_MANAGED_SUPERVISOR_REPORT_BYTES = 64,
  SAGR_MANAGED_SUPERVISOR_MAX_GRACE_MS = 60000
};

/* Private, host-local startup record. This is not part of the public runtime
 * ABI or the runtime-gem5 wire protocol. */
typedef struct sagr_managed_supervisor_report {
  uint64_t magic;
  uint32_t version;
  int32_t error_number;
  int64_t daemon_pid;
  uint8_t reserved[40];
} sagr_managed_supervisor_report_t;

_Static_assert(sizeof(sagr_managed_supervisor_report_t) ==
                   SAGR_MANAGED_SUPERVISOR_REPORT_BYTES,
               "managed supervisor report size changed");

#endif
