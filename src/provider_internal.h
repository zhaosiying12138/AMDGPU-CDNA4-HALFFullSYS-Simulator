/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_PROVIDER_INTERNAL_H
#define SELF_AMDGPU_RUNTIME_PROVIDER_INTERNAL_H

#include <stdatomic.h>
#include <sys/types.h>

#include <self_amdgpu_runtime/provider.h>

/* Private bridge used by the typed shim; callers never receive the instance
 * pointer and it is never serialized in a KMT envelope. */
struct sagr_provider {
  uint32_t magic;
  uint32_t reserved0;
  sagr_instance_t instance;
  sagr_managed_session_t managed_session;
  pid_t owner_pid;
  sagr_instance_info_t transport_info;
  atomic_uint_fast64_t kmt_operation_sequence;
};

sagr_instance_t sagr_provider_transport_instance(sagr_provider_t *provider);
const sagr_instance_info_t *sagr_provider_transport_info(
    const sagr_provider_t *provider);
int sagr_provider_is_valid(const sagr_provider_t *provider);
uint64_t sagr_provider_next_kmt_sequence(sagr_provider_t *provider);

#endif
