/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <self_amdgpu_runtime/runtime.h>

static int expect_string(const char *actual, const char *expected) {
  if (actual == NULL || strcmp(actual, expected) != 0) {
    fprintf(stderr, "expected '%s', got '%s'\n", expected,
            actual == NULL ? "(null)" : actual);
    return 1;
  }
  return 0;
}

int main(void) {
  int failures = 0;
  const uint32_t abi_version = sagr_abi_version();

  if (abi_version != SAGR_ABI_VERSION ||
      SAGR_ABI_VERSION_DECODE_MAJOR(abi_version) != SAGR_ABI_VERSION_MAJOR ||
      SAGR_ABI_VERSION_DECODE_MINOR(abi_version) != SAGR_ABI_VERSION_MINOR) {
    fprintf(stderr, "unexpected ABI version: 0x%08x\n", abi_version);
    ++failures;
  }

  failures += expect_string(sagr_version_string(), SAGR_VERSION_STRING);
  failures += expect_string(sagr_status_string(SAGR_STATUS_SUCCESS), "success");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_INVALID_ARGUMENT), "invalid argument");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_NOT_SUPPORTED), "not supported");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_INTERNAL_ERROR), "internal error");
  failures += expect_string(sagr_status_string(INT32_C(12345)), "unknown status");

  return failures == 0 ? 0 : 1;
}
