/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/runtime.h>

#include "transport_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CTest treats this return value as an explicit environment-gated skip. */
#define SAGR_ENDPOINT_TEST_SKIP 77

static int endpoint_unavailable(sagr_status_t status) {
  return status == SAGR_STATUS_ENDPOINT_NOT_FOUND ||
         status == SAGR_STATUS_UNAVAILABLE ||
         status == SAGR_STATUS_TIMED_OUT ||
         status == SAGR_STATUS_CONNECTION_LOST;
}

static uint64_t generic_dependency_mask(void) {
  return SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK |
         SAGR_CAPABILITY_MEMORY_MASK | SAGR_CAPABILITY_SIGNAL_MASK |
         SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK |
         SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
}

static int open_options_for_generic(
    sagr_instance_open_options_t *options) {
  const sagr_status_t status = sagr_instance_open_options_init(
      options, (uint32_t)sizeof(*options));
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic endpoint probe: options init failed: %s\n",
            sagr_status_string(status));
    return 1;
  }
  /* GENERIC_DISPATCH_V2 is a required selection here.  The runtime API
   * intentionally does not model optional capability offers; dependencies
   * are required as mandated by the v2 handshake contract. */
  options->offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |=
      generic_dependency_mask();
  options->required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |=
      generic_dependency_mask();
  return 0;
}

static int open_options_for_baseline(
    sagr_instance_open_options_t *options) {
  const sagr_status_t status = sagr_instance_open_options_init(
      options, (uint32_t)sizeof(*options));
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic endpoint probe: baseline options init failed: %s\n",
            sagr_status_string(status));
    return 1;
  }
  return 0;
}

static int close_instance(sagr_instance_t *instance) {
  const sagr_status_t status = sagr_instance_close(instance);
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic endpoint probe: instance close failed: %s\n",
            sagr_status_string(status));
    return 1;
  }
  return 0;
}

static void print_open_error(const char *label, sagr_status_t status,
                             const sagr_error_info_t *error) {
  fprintf(stderr,
          "generic endpoint probe: %s: status=%s wire_status=%d errno=%d message=%s\n",
          label, sagr_status_string(status), error->wire_status,
          error->native_errno, error->message[0] == '\0' ? "(none)"
                                                           : error->message);
}

int main(void) {
  const char *endpoint = getenv("SAGR_GENERIC_BRIDGE_ENDPOINT");
  sagr_instance_open_options_t generic_options;
  sagr_instance_open_options_t baseline_options;
  sagr_instance_t instance = NULL;
  sagr_instance_info_t info;
  sagr_error_info_t error;
  sagr_status_t status;
  const uint64_t generic_mask = SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
  const uint64_t dependency_mask = generic_dependency_mask();

  if (endpoint == NULL || endpoint[0] == '\0') {
    fprintf(stderr,
            "SKIP: SAGR_GENERIC_BRIDGE_ENDPOINT is unset; no daemon endpoint was probed\n");
    return SAGR_ENDPOINT_TEST_SKIP;
  }

  if (open_options_for_generic(&generic_options) != 0) {
    return 1;
  }
  status = sagr_instance_open(endpoint, &generic_options, &instance, &error,
                              (uint32_t)sizeof(error));
  if (status == SAGR_STATUS_CAPABILITY_MISMATCH &&
      error.wire_status == SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY) {
    /* A daemon that is reachable but has not advertised bit 8 is the
     * expected CP24 fail-closed boundary.  Retry the ordinary handshake so
     * this result is distinguished from an unavailable endpoint. */
    fprintf(stderr,
            "generic endpoint probe: endpoint reachable; generic capability was canonically rejected\n");
    if (open_options_for_baseline(&baseline_options) != 0) {
      return 1;
    }
    status = sagr_instance_open(endpoint, &baseline_options, &instance, &error,
                                (uint32_t)sizeof(error));
    if (endpoint_unavailable(status)) {
      print_open_error("baseline handshake after capability rejection",
                       status, &error);
      return SAGR_ENDPOINT_TEST_SKIP;
    }
    if (status != SAGR_STATUS_SUCCESS) {
      print_open_error("baseline handshake after capability rejection", status,
                       &error);
      return 1;
    }
    memset(&info, 0, sizeof(info));
    status = sagr_instance_get_info(instance, &info, (uint32_t)sizeof(info));
    if (status != SAGR_STATUS_SUCCESS) {
      fprintf(stderr, "generic endpoint probe: get_info failed: %s\n",
              sagr_status_string(status));
      (void)close_instance(&instance);
      return 1;
    }
    if ((info.negotiated_capabilities[SAGR_CAPABILITY_GENERIC_DISPATCH_WORD] &
         generic_mask) != 0U) {
      fprintf(stderr,
              "generic endpoint probe: baseline handshake unexpectedly selected generic capability\n");
      (void)close_instance(&instance);
      return 1;
    }
    if (close_instance(&instance) != 0) {
      return 1;
    }
    printf("{\"handshake\":true,\"generic_capability_selected\":false,"
           "\"canonical_unsupported\":true,\"execution\":false,"
           "\"launcher\":false}\n");
    return 0;
  }

  if (endpoint_unavailable(status)) {
    print_open_error("generic handshake", status, &error);
    fprintf(stderr,
            "SKIP: configured endpoint is unavailable; no daemon lifecycle was claimed\n");
    return SAGR_ENDPOINT_TEST_SKIP;
  }
  if (status != SAGR_STATUS_SUCCESS || instance == NULL) {
    print_open_error("generic handshake", status, &error);
    return 1;
  }

  memset(&info, 0, sizeof(info));
  status = sagr_instance_get_info(instance, &info, (uint32_t)sizeof(info));
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic endpoint probe: get_info failed: %s\n",
            sagr_status_string(status));
    (void)close_instance(&instance);
    return 1;
  }
  if ((info.negotiated_capabilities[SAGR_CAPABILITY_GENERIC_DISPATCH_WORD] &
       dependency_mask) != dependency_mask) {
    fprintf(stderr,
            "generic endpoint probe: successful handshake did not select bit 8 and all dependencies\n");
    (void)close_instance(&instance);
    return 1;
  }
  if (close_instance(&instance) != 0) {
    return 1;
  }
  printf("{\"handshake\":true,\"generic_capability_selected\":true,"
         "\"canonical_unsupported\":false,\"execution\":false,"
         "\"launcher\":false}\n");
  return 0;
}
