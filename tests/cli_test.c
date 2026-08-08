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
  queue_cli_options_t queue_options;
  memory_cli_options_t memory_options;
  signal_cli_options_t signal_options;
  dispatch_cli_options_t dispatch_options;
  const char *endpoint = NULL;
  uint64_t hold_ms = 0;
  memset(&queue_options, 0, sizeof(queue_options));
  memset(&memory_options, 0, sizeof(memory_options));
  memset(&signal_options, 0, sizeof(signal_options));
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  if (parse_arguments((int)(sizeof(arguments) / sizeof(arguments[0])),
                      arguments, &endpoint, &options, &hold_ms,
                      &queue_options, &memory_options, &signal_options,
                      &dispatch_options) != 0 ||
      endpoint == NULL || strcmp(endpoint, "/tmp/gemsim.sock") != 0 ||
      options.minimum_version_major != 0 ||
      options.minimum_version_minor != 9 ||
      options.maximum_version_major != 1 ||
      options.maximum_version_minor != 1 ||
      options.open_timeout_ns != UINT64_C(123000000) || hold_ms != 25 ||
      options.offered_capabilities[3] != (UINT64_C(1) << 63) ||
      options.required_capabilities[3] != (UINT64_C(1) << 63) ||
      options.offered_capabilities[0] != SAGR_CAPABILITY_TOPOLOGY_MASK ||
      options.required_capabilities[0] != SAGR_CAPABILITY_TOPOLOGY_MASK ||
      queue_options.enabled != 0 || memory_options.enabled != 0 ||
      memory_options.alignment != SAGR_MEMORY_ALIGNMENT_4K ||
      dispatch_options.enabled != 0) {
    fprintf(stderr, "complete CLI option parse failed\n");
    return 1;
  }
  return 0;
}

static int expect_invalid(char **arguments, size_t argument_count) {
  sagr_instance_open_options_t options;
  queue_cli_options_t queue_options;
  memory_cli_options_t memory_options;
  signal_cli_options_t signal_options;
  dispatch_cli_options_t dispatch_options;
  const char *endpoint = NULL;
  uint64_t hold_ms = 0;
  memset(&queue_options, 0, sizeof(queue_options));
  memset(&memory_options, 0, sizeof(memory_options));
  memset(&signal_options, 0, sizeof(signal_options));
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  return parse_arguments((int)argument_count, arguments, &endpoint, &options,
                         &hold_ms, &queue_options, &memory_options,
                         &signal_options, &dispatch_options) == -1
             ? 0
             : 1;
}

static int test_queue_parse(void) {
  char *arguments[] = {
      (char *)"sagr-handshake",
      (char *)"--endpoint",
      (char *)"/tmp/gemsim.sock",
      (char *)"--queue-depth",
      (char *)"8",
      (char *)"--doorbells",
      (char *)"3",
      (char *)"--command-kind",
      (char *)"1",
  };
  char *partial[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--queue-depth", (char *)"4"};
  char *bad_depth[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--queue-depth", (char *)"65", (char *)"--doorbells",
      (char *)"1", (char *)"--command-kind", (char *)"0"};
  char *bad_kind[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--queue-depth", (char *)"4", (char *)"--doorbells",
      (char *)"1", (char *)"--command-kind", (char *)"3"};
  sagr_instance_open_options_t options;
  queue_cli_options_t queue_options;
  memory_cli_options_t memory_options;
  signal_cli_options_t signal_options;
  dispatch_cli_options_t dispatch_options;
  const char *endpoint = NULL;
  uint64_t hold_ms = 0;
  memset(&queue_options, 0, sizeof(queue_options));
  memset(&memory_options, 0, sizeof(memory_options));
  memset(&signal_options, 0, sizeof(signal_options));
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  if (parse_arguments((int)(sizeof(arguments) / sizeof(arguments[0])),
                      arguments, &endpoint, &options, &hold_ms,
                      &queue_options, &memory_options, &signal_options,
                      &dispatch_options) != 0 ||
      queue_options.enabled != 1 || queue_options.depth != 8 ||
      queue_options.doorbells != 3 ||
      queue_options.command_kind != SAGR_QUEUE_COMMAND_CONTROL_TEST ||
      (options.offered_capabilities[0] & SAGR_CAPABILITY_QUEUE_MASK) == 0 ||
      (options.required_capabilities[0] & SAGR_CAPABILITY_QUEUE_MASK) == 0 ||
      expect_invalid(partial, sizeof(partial) / sizeof(partial[0])) != 0 ||
      expect_invalid(bad_depth, sizeof(bad_depth) / sizeof(bad_depth[0])) !=
          0 ||
      expect_invalid(bad_kind, sizeof(bad_kind) / sizeof(bad_kind[0])) != 0) {
    fprintf(stderr, "queue CLI option parse failed\n");
    return 1;
  }
  arguments[8] = (char *)"2";
  endpoint = NULL;
  hold_ms = 0;
  memset(&queue_options, 0, sizeof(queue_options));
  memset(&memory_options, 0, sizeof(memory_options));
  memset(&signal_options, 0, sizeof(signal_options));
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  if (parse_arguments((int)(sizeof(arguments) / sizeof(arguments[0])),
                      arguments, &endpoint, &options, &hold_ms,
                      &queue_options, &memory_options, &signal_options,
                      &dispatch_options) != 0 ||
      queue_options.command_kind != SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST) {
    fprintf(stderr, "CONTROL_ERROR_TEST CLI option parse failed\n");
    return 1;
  }
  return 0;
}

static int test_memory_parse(void) {
  char *arguments[] = {
      (char *)"sagr-handshake", (char *)"--endpoint",
      (char *)"/tmp/gemsim.sock", (char *)"--memory-bytes",
      (char *)"65536", (char *)"--memory-alignment", (char *)"65536",
      (char *)"--memory-reuse"};
  char *default_alignment[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--memory-bytes", (char *)"1"};
  char *alignment_without_memory[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--memory-alignment", (char *)"4096"};
  char *reuse_without_memory[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--memory-reuse"};
  char *zero_bytes[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--memory-bytes", (char *)"0"};
  char *oversize[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--memory-bytes", (char *)"16777217"};
  char *bad_alignment[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--memory-bytes", (char *)"1",
      (char *)"--memory-alignment", (char *)"8192"};
  sagr_instance_open_options_t options;
  queue_cli_options_t queue_options;
  memory_cli_options_t memory_options;
  signal_cli_options_t signal_options;
  dispatch_cli_options_t dispatch_options;
  const char *endpoint = NULL;
  uint64_t hold_ms = 0;

  memset(&queue_options, 0, sizeof(queue_options));
  memset(&memory_options, 0, sizeof(memory_options));
  memset(&signal_options, 0, sizeof(signal_options));
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  if (parse_arguments((int)(sizeof(arguments) / sizeof(arguments[0])),
                      arguments, &endpoint, &options, &hold_ms,
                      &queue_options, &memory_options, &signal_options,
                      &dispatch_options) != 0 ||
      memory_options.enabled != 1 || memory_options.reuse != 1 ||
      memory_options.bytes != UINT64_C(65536) ||
      memory_options.alignment != SAGR_MEMORY_ALIGNMENT_64K ||
      (options.offered_capabilities[SAGR_CAPABILITY_MEMORY_WORD] &
       SAGR_CAPABILITY_MEMORY_MASK) == 0 ||
      (options.required_capabilities[SAGR_CAPABILITY_MEMORY_WORD] &
       SAGR_CAPABILITY_MEMORY_MASK) == 0) {
    fprintf(stderr, "memory CLI option parse failed\n");
    return 1;
  }

  endpoint = NULL;
  hold_ms = 0;
  memset(&queue_options, 0, sizeof(queue_options));
  memset(&memory_options, 0, sizeof(memory_options));
  memset(&signal_options, 0, sizeof(signal_options));
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  if (parse_arguments(
          (int)(sizeof(default_alignment) / sizeof(default_alignment[0])),
          default_alignment, &endpoint, &options, &hold_ms, &queue_options,
          &memory_options, &signal_options, &dispatch_options) != 0 ||
      memory_options.enabled != 1 || memory_options.reuse != 0 ||
      memory_options.bytes != UINT64_C(1) ||
      memory_options.alignment != SAGR_MEMORY_ALIGNMENT_4K ||
      expect_invalid(alignment_without_memory,
                     sizeof(alignment_without_memory) /
                         sizeof(alignment_without_memory[0])) != 0 ||
      expect_invalid(reuse_without_memory,
                     sizeof(reuse_without_memory) /
                         sizeof(reuse_without_memory[0])) != 0 ||
      expect_invalid(zero_bytes,
                     sizeof(zero_bytes) / sizeof(zero_bytes[0])) != 0 ||
      expect_invalid(oversize, sizeof(oversize) / sizeof(oversize[0])) != 0 ||
      expect_invalid(bad_alignment,
                     sizeof(bad_alignment) / sizeof(bad_alignment[0])) != 0) {
    fprintf(stderr, "memory CLI default or rejection parse failed\n");
    return 1;
  }
  return 0;
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

static int test_dispatch_parse(void) {
  char *arguments[] = {
      (char *)"sagr-handshake", (char *)"--endpoint", (char *)"/tmp/gemsim.sock",
      (char *)"--dispatch-fixture", (char *)"gfx950-xor-u8-v1"};
  char *wrong_fixture[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--dispatch-fixture", (char *)"gfx950-store-u32-v1"};
  char *mixed_mode[] = {
      (char *)"tool", (char *)"--endpoint", (char *)"/x",
      (char *)"--dispatch-fixture", (char *)"gfx950-xor-u8-v1",
      (char *)"--queue-depth", (char *)"1", (char *)"--doorbells",
      (char *)"0", (char *)"--command-kind", (char *)"0"};
  sagr_instance_open_options_t options;
  queue_cli_options_t queue_options;
  memory_cli_options_t memory_options;
  signal_cli_options_t signal_options;
  dispatch_cli_options_t dispatch_options;
  const char *endpoint = NULL;
  uint64_t hold_ms = 0;
  memset(&queue_options, 0, sizeof(queue_options));
  memset(&memory_options, 0, sizeof(memory_options));
  memset(&signal_options, 0, sizeof(signal_options));
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  if (parse_arguments((int)(sizeof(arguments) / sizeof(arguments[0])),
                      arguments, &endpoint, &options, &hold_ms,
                      &queue_options, &memory_options, &signal_options,
                      &dispatch_options) != 0 ||
      dispatch_options.enabled != 1 || queue_options.enabled != 0 ||
      memory_options.enabled != 0 || signal_options.enabled != 0 ||
      options.offered_capabilities[0] != UINT64_C(0x1f) ||
      options.required_capabilities[0] != UINT64_C(0x1f) ||
      expect_invalid(wrong_fixture,
                     sizeof(wrong_fixture) / sizeof(wrong_fixture[0])) != 0 ||
      expect_invalid(mixed_mode, sizeof(mixed_mode) / sizeof(mixed_mode[0])) !=
          0) {
    fprintf(stderr, "dispatch CLI option parse failed\n");
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
  failures += test_queue_parse();
  failures += test_memory_parse();
  failures += test_dispatch_parse();
  failures += test_monotonic_hold();
  return failures == 0 ? 0 : 1;
}
