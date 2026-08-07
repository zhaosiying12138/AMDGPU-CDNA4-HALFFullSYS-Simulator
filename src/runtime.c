/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/runtime.h>

uint32_t sagr_abi_version(void) {
  return SAGR_ABI_VERSION;
}

const char *sagr_version_string(void) {
  return SAGR_VERSION_STRING;
}

const char *sagr_status_string(sagr_status_t status) {
  switch (status) {
    case SAGR_STATUS_SUCCESS:
      return "success";
    case SAGR_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case SAGR_STATUS_NOT_SUPPORTED:
      return "not supported";
    case SAGR_STATUS_INTERNAL_ERROR:
      return "internal error";
    default:
      return "unknown status";
  }
}
