/* SPDX-License-Identifier: GPL-3.0-or-later */

/* Reuse the exact product example so the CTest gate cannot drift from the
 * user-facing executable. */
#define main sagr_opencl_vecadd_example_main
#include "../examples/opencl/vecadd_host.c"
#undef main

#ifndef SAGR_OPENCL_E2E_DEFAULT
#define SAGR_OPENCL_E2E_DEFAULT 0
#endif

#define SAGR_OPENCL_E2E_SKIP 77

static int e2e_is_enabled(void) {
  const char *value = getenv("SAGR_OPENCL_E2E");

  if (value == NULL || value[0] == '\0') {
    return SAGR_OPENCL_E2E_DEFAULT != 0;
  }
  return strcmp(value, "1") == 0 || strcmp(value, "true") == 0 ||
         strcmp(value, "yes") == 0 || strcmp(value, "on") == 0;
}

int main(int argc, char **argv) {
  if (!e2e_is_enabled()) {
    printf("{\"schema\":\"self-amdgpu-runtime.opencl-vecadd-e2e.v1\","
           "\"skipped\":true,"
           "\"reason\":\"set SAGR_OPENCL_E2E=1 when gem5 is available\"}\n");
    return SAGR_OPENCL_E2E_SKIP;
  }
  return sagr_opencl_vecadd_example_main(argc, argv);
}
