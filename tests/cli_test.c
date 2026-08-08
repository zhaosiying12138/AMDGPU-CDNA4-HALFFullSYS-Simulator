/* SPDX-License-Identifier: GPL-3.0-or-later */

#define main sagr_handshake_entry
#include "../tools/sagr-handshake.c"
#undef main

static int test_complete_parse(void) {
  char *arguments[] = {
      (char *)"sagr-handshake",
      (char *)"--endpoint",
      (char *)"/tmp/gemsim.sock",
      (char *)"--min-version",
      (char *)"0.9",
      (char *)"--max-version",
      (char *)"1.1",
      (char *)"--offer-cap-bit",
      (char *)"255",
      (char *)"--require-cap-bit",
      (char *)"255",
      (char *)"--timeout-ms",
      (char *)"123",
      (char *)"--hold-ms",
      (char *)"25",
  };
  sagr_instance_open_options_t options;
  const char *endpoint = NULL;
  uint64_t hold_ms = 0;
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  if (parse_arguments((int)(sizeof(arguments) / sizeof(arguments[0])),
                      arguments, &endpoint, &options, &hold_ms) != 0 ||
      endpoint == NULL || strcmp(endpoint, "/tmp/gemsim.sock") != 0 ||
      options.minimum_version_major != 0 ||
      options.minimum_version_minor != 9 ||
      options.maximum_version_major != 1 ||
      options.maximum_version_minor != 1 ||
      options.open_timeout_ns != UINT64_C(123000000) || hold_ms != 25 ||
      options.offered_capabilities[3] != (UINT64_C(1) << 63) ||
      options.required_capabilities[3] != (UINT64_C(1) << 63) ||
      options.offered_capabilities[0] != SAGR_CAPABILITY_TOPOLOGY_MASK ||
      options.required_capabilities[0] != SAGR_CAPABILITY_TOPOLOGY_MASK) {
    fprintf(stderr, "complete CLI option parse failed\n");
    return 1;
  }
  return 0;
}

static int expect_invalid(char **arguments, size_t argument_count) {
  sagr_instance_open_options_t options;
  const char *endpoint = NULL;
  uint64_t hold_ms = 0;
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  return parse_arguments((int)argument_count, arguments, &endpoint, &options,
                         &hold_ms) == -1
             ? 0
             : 1;
}

static int test_invalid_parse(void) {
  char *bad_cap[] = {(char *)"tool", (char *)"--endpoint", (char *)"/x",
                     (char *)"--offer-cap-bit", (char *)"256"};
  char *bad_version[] = {(char *)"tool", (char *)"--endpoint", (char *)"/x",
                         (char *)"--min-version", (char *)"1"};
  char *bad_hold[] = {(char *)"tool", (char *)"--endpoint", (char *)"/x",
                      (char *)"--hold-ms", (char *)"-1"};
  char *bad_space[] = {(char *)"tool", (char *)"--endpoint", (char *)"/x",
                       (char *)"--timeout-ms", (char *)" 1"};
  char *bad_uuid_hyphens[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--expected-daemon-uuid",
      (char *)"-00112233445566778899aabbccddeeff"};
  char *bad_version_space[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--min-version", (char *)"1. 0"};
  char *partial_topology[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--expected-job-uuid",
      (char *)"102132435465768798a9bacbdcedfe0f"};
  const int failures =
      expect_invalid(bad_cap, sizeof(bad_cap) / sizeof(bad_cap[0])) +
      expect_invalid(bad_version,
                     sizeof(bad_version) / sizeof(bad_version[0])) +
      expect_invalid(bad_hold, sizeof(bad_hold) / sizeof(bad_hold[0])) +
      expect_invalid(bad_space, sizeof(bad_space) / sizeof(bad_space[0])) +
      expect_invalid(bad_uuid_hyphens,
                     sizeof(bad_uuid_hyphens) /
                         sizeof(bad_uuid_hyphens[0])) +
      expect_invalid(bad_version_space,
                     sizeof(bad_version_space) /
                         sizeof(bad_version_space[0])) +
      expect_invalid(partial_topology,
                     sizeof(partial_topology) / sizeof(partial_topology[0]));
  if (failures != 0) {
    fprintf(stderr, "invalid CLI option was accepted\n");
    return 1;
  }
  return 0;
}

static int test_monotonic_hold(void) {
  struct timespec before;
  struct timespec after;
  int64_t elapsed_ns;
  if (clock_gettime(CLOCK_MONOTONIC, &before) != 0 ||
      hold_connection(20) != 0 ||
      clock_gettime(CLOCK_MONOTONIC, &after) != 0) {
    fprintf(stderr, "monotonic hold failed\n");
    return 1;
  }
  elapsed_ns = ((int64_t)after.tv_sec - (int64_t)before.tv_sec) *
                   INT64_C(1000000000) +
               (int64_t)after.tv_nsec - (int64_t)before.tv_nsec;
  if (elapsed_ns < INT64_C(15000000)) {
    fprintf(stderr, "monotonic hold returned too early\n");
    return 1;
  }
  return 0;
}

int main(void) {
  int failures = 0;
  failures += test_complete_parse();
  failures += test_invalid_parse();
  failures += test_monotonic_hold();
  return failures == 0 ? 0 : 1;
}
