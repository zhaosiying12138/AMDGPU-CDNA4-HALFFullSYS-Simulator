/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_RUNTIME_H
#define SELF_AMDGPU_RUNTIME_RUNTIME_H

#include <stdint.h>

#include <self_amdgpu_runtime/export.h>
#include <self_amdgpu_runtime/version.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t sagr_status_t;

enum {
  SAGR_STATUS_SUCCESS = 0,
  SAGR_STATUS_INVALID_ARGUMENT = 1,
  SAGR_STATUS_NOT_SUPPORTED = 2,
  SAGR_STATUS_INTERNAL_ERROR = 3
};

SAGR_API uint32_t sagr_abi_version(void);
SAGR_API const char *sagr_version_string(void);
SAGR_API const char *sagr_status_string(sagr_status_t status);

#ifdef __cplusplus
}
#endif

#endif
