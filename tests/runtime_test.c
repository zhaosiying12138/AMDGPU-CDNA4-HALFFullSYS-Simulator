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

static int expect_options_defaults(void) {
  sagr_instance_open_options_t options;
  const sagr_status_t status =
      sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));

  if (status != SAGR_STATUS_SUCCESS ||
      options.struct_size != (uint32_t)sizeof(options) ||
      options.minimum_version_major != 1 ||
      options.minimum_version_minor != 0 ||
      options.maximum_version_major != 1 ||
      options.maximum_version_minor != 0 ||
      options.open_timeout_ns != SAGR_DEFAULT_OPEN_TIMEOUT_NS ||
      options.offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] !=
          SAGR_CAPABILITY_TOPOLOGY_MASK ||
      options.required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] !=
          SAGR_CAPABILITY_TOPOLOGY_MASK ||
      options.expected_epoch != 0 ||
      options.expected_rank != SAGR_INSTANCE_RANK_WILDCARD ||
      options.expected_world_size != 0 || options.absolute_deadline_ns != 0 ||
      options.cancel_fd != -1 || options.reserved0 != 0) {
    fprintf(stderr, "unexpected instance option defaults\n");
    return 1;
  }

  if (SAGR_CAPABILITY_GENERIC_EXECUTION_WORD != UINT32_C(0) ||
      SAGR_CAPABILITY_GENERIC_EXECUTION_MASK != (UINT64_C(1) << 9) ||
      SAGR_CAPABILITY_GENERIC_DISPATCH_MASK != (UINT64_C(1) << 8)) {
    fprintf(stderr, "generic execution capability numbering drifted\n");
    return 1;
  }

  if (sagr_instance_open_options_init(NULL, (uint32_t)sizeof(options)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_instance_open_options_init(&options, (uint32_t)sizeof(options) - 1) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      options.struct_size != (uint32_t)sizeof(options)) {
    fprintf(stderr, "unexpected instance option validation\n");
    return 1;
  }

  return 0;
}

static int expect_queue_option_defaults(void) {
  sagr_queue_create_options_t create_options;
  sagr_queue_operation_options_t operation_options;
  if (sagr_queue_create_options_init(
          &create_options, (uint32_t)sizeof(create_options)) !=
          SAGR_STATUS_SUCCESS ||
      create_options.struct_size != (uint32_t)sizeof(create_options) ||
      create_options.flags != 0 || create_options.depth != 1 ||
      create_options.reserved0 != 0 ||
      sagr_queue_operation_options_init(
          &operation_options, (uint32_t)sizeof(operation_options)) !=
          SAGR_STATUS_SUCCESS ||
      operation_options.struct_size != (uint32_t)sizeof(operation_options) ||
      operation_options.flags != 0 ||
      operation_options.timeout_ns != SAGR_DEFAULT_OPEN_TIMEOUT_NS ||
      operation_options.absolute_deadline_ns != 0 ||
      operation_options.cancel_fd != -1 ||
      operation_options.reserved0 != 0) {
    fprintf(stderr, "unexpected queue option defaults\n");
    return 1;
  }
  if (sagr_queue_create_options_init(
          NULL, (uint32_t)sizeof(create_options)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_queue_create_options_init(
          &create_options, (uint32_t)sizeof(create_options) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      create_options.struct_size != (uint32_t)sizeof(create_options) ||
      sagr_queue_operation_options_init(
          NULL, (uint32_t)sizeof(operation_options)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_queue_operation_options_init(
          &operation_options, (uint32_t)sizeof(operation_options) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      operation_options.struct_size != (uint32_t)sizeof(operation_options)) {
    fprintf(stderr, "unexpected queue option validation\n");
    return 1;
  }
  return 0;
}

static int expect_memory_option_defaults(void) {
  sagr_memory_allocate_options_t allocate_options;
  sagr_memory_operation_options_t operation_options;
  if (sagr_memory_allocate_options_init(
          &allocate_options, (uint32_t)sizeof(allocate_options)) !=
          SAGR_STATUS_SUCCESS ||
      allocate_options.struct_size != (uint32_t)sizeof(allocate_options) ||
      allocate_options.flags != 0 || allocate_options.size_bytes != 0 ||
      allocate_options.alignment_bytes != SAGR_MEMORY_ALIGNMENT_4K ||
      sagr_memory_operation_options_init(
          &operation_options, (uint32_t)sizeof(operation_options)) !=
          SAGR_STATUS_SUCCESS ||
      operation_options.struct_size != (uint32_t)sizeof(operation_options) ||
      operation_options.flags != 0 ||
      operation_options.timeout_ns != SAGR_DEFAULT_OPEN_TIMEOUT_NS ||
      operation_options.absolute_deadline_ns != 0 ||
      operation_options.cancel_fd != -1 || operation_options.reserved0 != 0) {
    fprintf(stderr, "unexpected memory option defaults\n");
    return 1;
  }
  if (sagr_memory_allocate_options_init(
          NULL, (uint32_t)sizeof(allocate_options)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_memory_allocate_options_init(
          &allocate_options, (uint32_t)sizeof(allocate_options) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      allocate_options.struct_size != (uint32_t)sizeof(allocate_options) ||
      sagr_memory_operation_options_init(
          NULL, (uint32_t)sizeof(operation_options)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_memory_operation_options_init(
          &operation_options, (uint32_t)sizeof(operation_options) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      operation_options.struct_size != (uint32_t)sizeof(operation_options)) {
    fprintf(stderr, "unexpected memory option validation\n");
    return 1;
  }
  return 0;
}

static int expect_signal_option_defaults(void) {
  sagr_signal_create_options_t create_options;
  sagr_signal_operation_options_t operation_options;
  if (sagr_signal_create_options_init(
          &create_options, (uint32_t)sizeof(create_options)) !=
          SAGR_STATUS_SUCCESS ||
      create_options.struct_size != (uint32_t)sizeof(create_options) ||
      create_options.flags != 0 || create_options.initial_value != 0 ||
      sagr_signal_operation_options_init(
          &operation_options, (uint32_t)sizeof(operation_options)) !=
          SAGR_STATUS_SUCCESS ||
      operation_options.struct_size != (uint32_t)sizeof(operation_options) ||
      operation_options.flags != 0 ||
      operation_options.timeout_ns != SAGR_DEFAULT_OPEN_TIMEOUT_NS ||
      operation_options.absolute_deadline_ns != 0 ||
      operation_options.cancel_fd != -1 || operation_options.reserved0 != 0) {
    fprintf(stderr, "unexpected signal option defaults\n");
    return 1;
  }
  if (sagr_signal_create_options_init(
          NULL, (uint32_t)sizeof(create_options)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_signal_create_options_init(
          &create_options, (uint32_t)sizeof(create_options) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      create_options.struct_size != (uint32_t)sizeof(create_options) ||
      sagr_signal_operation_options_init(
          NULL, (uint32_t)sizeof(operation_options)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_signal_operation_options_init(
          &operation_options, (uint32_t)sizeof(operation_options) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      operation_options.struct_size != (uint32_t)sizeof(operation_options)) {
    fprintf(stderr, "unexpected signal option validation\n");
    return 1;
  }
  return 0;
}

static int expect_dispatch_option_defaults(void) {
  sagr_pinned_dispatch_options_t options;
  if (sagr_pinned_dispatch_options_init(
          &options, (uint32_t)sizeof(options)) != SAGR_STATUS_SUCCESS ||
      options.struct_size != (uint32_t)sizeof(options) ||
      options.flags != 0 ||
      options.fixture_id != SAGR_DISPATCH_FIXTURE_GFX950_XOR_U8_V1) {
    fprintf(stderr, "unexpected pinned dispatch option defaults\n");
    return 1;
  }
  if (sagr_pinned_dispatch_options_init(NULL, (uint32_t)sizeof(options)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_pinned_dispatch_options_init(
          &options, (uint32_t)sizeof(options) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      options.struct_size != (uint32_t)sizeof(options)) {
    fprintf(stderr, "unexpected pinned dispatch option validation\n");
    return 1;
  }
  return 0;
}

static int expect_managed_option_defaults(void) {
  sagr_managed_session_options_t session;
  sagr_managed_session_options_v2_t exact_session;
  sagr_managed_launch_options_t launch;
  if (sagr_managed_session_options_init(
          &session, (uint32_t)sizeof(session)) != SAGR_STATUS_SUCCESS ||
      session.struct_size != (uint32_t)sizeof(session) ||
      session.version != SAGR_MANAGED_RUNTIME_API_VERSION ||
      session.flags != 0U ||
      session.queue_depth != SAGR_MANAGED_DEFAULT_QUEUE_DEPTH ||
      session.startup_timeout_ns !=
          SAGR_MANAGED_DEFAULT_STARTUP_TIMEOUT_NS ||
      session.operation_timeout_ns !=
          SAGR_MANAGED_DEFAULT_OPERATION_TIMEOUT_NS ||
      session.run_timeout_ns != SAGR_MANAGED_DEFAULT_RUN_TIMEOUT_NS ||
      session.startup_timeout_ns != UINT64_C(15000000000) ||
      session.operation_timeout_ns != UINT64_C(21600000000000) ||
      session.run_timeout_ns != UINT64_C(86400000000000) ||
      sagr_managed_session_options_v2_init(
          &exact_session, (uint32_t)sizeof(exact_session)) !=
          SAGR_STATUS_SUCCESS ||
      exact_session.struct_size != (uint32_t)sizeof(exact_session) ||
      exact_session.version != SAGR_MANAGED_SESSION_OPTIONS_V2_VERSION ||
      exact_session.flags != 0U || exact_session.epoch != 0U ||
      exact_session.rank != 0U || exact_session.world_size != 0U ||
      exact_session.queue_depth != SAGR_MANAGED_DEFAULT_QUEUE_DEPTH ||
      exact_session.startup_timeout_ns !=
          SAGR_MANAGED_DEFAULT_STARTUP_TIMEOUT_NS ||
      exact_session.operation_timeout_ns !=
          SAGR_MANAGED_DEFAULT_OPERATION_TIMEOUT_NS ||
      exact_session.run_timeout_ns !=
          SAGR_MANAGED_DEFAULT_RUN_TIMEOUT_NS ||
      sagr_managed_launch_options_init(
          &launch, (uint32_t)sizeof(launch)) != SAGR_STATUS_SUCCESS ||
      launch.struct_size != (uint32_t)sizeof(launch) ||
      launch.version != SAGR_MANAGED_RUNTIME_API_VERSION ||
      launch.grid_x != 64U || launch.grid_y != 1U || launch.grid_z != 1U ||
      launch.workgroup_x != 64U || launch.workgroup_y != 1U ||
      launch.workgroup_z != 1U || launch.num_warps != 1U ||
      launch.num_ctas != 1U || launch.wavefront_size != 64U) {
    fprintf(stderr, "unexpected managed API defaults\n");
    return 1;
  }
  session.queue_depth = 32U;
  session.startup_timeout_ns = UINT64_C(1000000000);
  session.operation_timeout_ns = UINT64_C(2000000000);
  session.run_timeout_ns = UINT64_C(3000000000);
  if (session.queue_depth != 32U ||
      session.startup_timeout_ns != UINT64_C(1000000000) ||
      session.operation_timeout_ns != UINT64_C(2000000000) ||
      session.run_timeout_ns != UINT64_C(3000000000)) {
    fprintf(stderr, "managed API caller overrides were not retained\n");
    return 1;
  }
  if (sagr_managed_session_options_init(NULL,
                                         (uint32_t)sizeof(session)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_managed_session_options_init(
          &session, (uint32_t)sizeof(session) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      session.struct_size != (uint32_t)sizeof(session) ||
      sagr_managed_session_options_v2_init(NULL,
                                            (uint32_t)sizeof(exact_session)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_managed_session_options_v2_init(
          &exact_session, (uint32_t)sizeof(exact_session) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      exact_session.struct_size != (uint32_t)sizeof(exact_session) ||
      sagr_managed_launch_options_init(NULL, (uint32_t)sizeof(launch)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_managed_launch_options_init(
          &launch, (uint32_t)sizeof(launch) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      launch.struct_size != (uint32_t)sizeof(launch)) {
    fprintf(stderr, "unexpected managed API option validation\n");
    return 1;
  }
  return 0;
}

static int expect_exact_managed_option_validation(void) {
  sagr_managed_session_options_v2_t options;
  sagr_managed_session_options_t legacy_options;
  sagr_managed_session_t session = NULL;
  sagr_error_info_t error;
  if (sagr_managed_session_options_init(
          &legacy_options, (uint32_t)sizeof(legacy_options)) !=
          SAGR_STATUS_SUCCESS) {
    return 1;
  }
  legacy_options.flags = UINT32_C(2);
  if (sagr_managed_session_open(
          &legacy_options, &session, NULL, 0U, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_INVALID_ARGUMENT ||
      session != NULL) {
    fprintf(stderr, "unknown managed-session v1 flag was accepted\n");
    return 1;
  }
  if (sagr_managed_session_options_v2_init(
          &options, (uint32_t)sizeof(options)) != SAGR_STATUS_SUCCESS ||
      sagr_managed_session_open_v2(
          &options, &session, NULL, 0U, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_INVALID_ARGUMENT ||
      session != NULL) {
    fprintf(stderr, "zero exact-topology identity was accepted\n");
    return 1;
  }

  options.epoch = 1U;
  options.world_size = 2U;
  options.rank = 2U;
  options.job_uuid[0] = 1U;
  if (sagr_managed_session_open_v2(
          &options, &session, NULL, 0U, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_INVALID_ARGUMENT ||
      session != NULL) {
    fprintf(stderr, "out-of-range exact-topology rank was accepted\n");
    return 1;
  }

  options.rank = 1U;
  options.flags = SAGR_MANAGED_SESSION_V2_FLAG_EXTERNAL_ENDPOINT;
  memcpy(options.endpoint, "relative.sock", sizeof("relative.sock"));
  if (sagr_managed_session_open_v2(
          &options, &session, NULL, 0U, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_INVALID_ARGUMENT ||
      session != NULL) {
    fprintf(stderr, "relative exact-topology endpoint was accepted\n");
    return 1;
  }

  memset(options.endpoint, 0, sizeof(options.endpoint));
  options.flags = 0U;
  options.endpoint[0] = (uint8_t)'/';
  if (sagr_managed_session_open_v2(
          &options, &session, NULL, 0U, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_INVALID_ARGUMENT ||
      session != NULL) {
    fprintf(stderr, "unselected exact-topology endpoint was accepted\n");
    return 1;
  }

  memset(options.endpoint, 0, sizeof(options.endpoint));
  options.flags = SAGR_MANAGED_SESSION_V2_FLAG_EXTERNAL_ENDPOINT |
                  SAGR_MANAGED_SESSION_V2_FLAG_PRIVATE_NAMESPACE;
  memcpy(options.endpoint, "/tmp/both.sock", sizeof("/tmp/both.sock"));
  if (sagr_managed_session_open_v2(
          &options, &session, NULL, 0U, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_INVALID_ARGUMENT ||
      session != NULL) {
    fprintf(stderr, "conflicting exact-topology endpoint modes were accepted\n");
    return 1;
  }
  return 0;
}

int main(void) {
  int failures = 0;
  const uint32_t abi_version = sagr_abi_version();

  _Static_assert(sizeof(sagr_instance_open_options_t) == 160,
                 "public options ABI size changed");
  _Static_assert(sizeof(sagr_instance_info_t) == 152,
                 "public instance info ABI size changed");
  _Static_assert(sizeof(sagr_error_info_t) == 160,
                 "public error info ABI size changed");
  _Static_assert(sizeof(sagr_queue_operation_options_t) == 48,
                 "public queue operation options ABI size changed");
  _Static_assert(sizeof(sagr_queue_create_options_t) == 32,
                 "public queue create options ABI size changed");
  _Static_assert(sizeof(sagr_queue_info_t) == 80,
                 "public queue info ABI size changed");
  _Static_assert(sizeof(sagr_queue_completion_t) == 80,
                 "public queue completion ABI size changed");
  _Static_assert(sizeof(sagr_memory_allocate_options_t) == 40,
                 "public memory allocate options ABI size changed");
  _Static_assert(sizeof(sagr_memory_operation_options_t) == 48,
                 "public memory operation options ABI size changed");
  _Static_assert(sizeof(sagr_memory_info_t) == 96,
                 "public memory info ABI size changed");
  _Static_assert(sizeof(sagr_signal_create_options_t) == 32,
                 "public signal create options ABI size changed");
  _Static_assert(sizeof(sagr_signal_operation_options_t) == 48,
                 "public signal operation options ABI size changed");
  _Static_assert(sizeof(sagr_signal_info_t) == 80,
                 "public signal info ABI size changed");
  _Static_assert(sizeof(sagr_signal_wait_result_t) == 88,
                 "public signal wait result ABI size changed");
  _Static_assert(sizeof(sagr_pinned_dispatch_options_t) == 32,
                 "public pinned dispatch options ABI size changed");
  _Static_assert(sizeof(sagr_dispatch_ticket_t) == 152,
                 "public dispatch ticket ABI size changed");
  _Static_assert(sizeof(sagr_dispatch_completion_t) == 184,
                 "public dispatch completion ABI size changed");
  _Static_assert(sizeof(sagr_managed_session_options_t) == 64,
                 "managed session options ABI size changed");
  _Static_assert(sizeof(sagr_managed_session_options_v2_t) == 200,
                 "managed session v2 options ABI size changed");
  _Static_assert(sizeof(sagr_managed_session_info_t) == 96,
                 "managed session info ABI size changed");
  _Static_assert(sizeof(sagr_managed_kernel_info_t) == 176,
                 "managed kernel info ABI size changed");
  _Static_assert(sizeof(sagr_managed_launch_options_t) == 76,
                 "managed launch options ABI size changed");
  _Static_assert(sizeof(sagr_generic_dispatch_completion_t) == 304,
                 "generic completion ABI size changed");

  if (abi_version != SAGR_ABI_VERSION ||
      strcmp(SAGR_VERSION_STRING, "0.8.0") != 0 ||
      SAGR_ABI_VERSION_DECODE_MAJOR(abi_version) != SAGR_ABI_VERSION_MAJOR ||
      SAGR_ABI_VERSION_DECODE_MINOR(abi_version) != SAGR_ABI_VERSION_MINOR ||
      SAGR_ABI_VERSION_MAJOR != 1 || SAGR_ABI_VERSION_MINOR != 8 ||
      SAGR_CAPABILITY_QUEUE_MASK != UINT64_C(2) ||
      SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST != UINT64_C(2) ||
      SAGR_CAPABILITY_MEMORY_MASK != UINT64_C(4) ||
      SAGR_MEMORY_MAX_TRANSFER_BYTES != UINT64_C(16777216) ||
      SAGR_CAPABILITY_SIGNAL_MASK != UINT64_C(8) ||
      SAGR_CAPABILITY_DISPATCH_MASK != UINT64_C(16) ||
      SAGR_DISPATCH_FIXTURE_GFX950_XOR_U8_V1 != UINT64_C(1) ||
      SAGR_DISPATCH_FIXED_IO_BYTES != UINT64_C(64) ||
      SAGR_DISPATCH_PACKET_CRC32C != UINT32_C(0x8a912d83) ||
      SAGR_DISPATCH_OUTPUT_CRC32C != UINT32_C(0x796671ec)) {
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
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_VERSION_MISMATCH), "version mismatch");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_CAPABILITY_MISMATCH),
      "capability mismatch");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_ENDPOINT_NOT_FOUND),
      "endpoint not found");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_INSTANCE_MISMATCH), "instance mismatch");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_TOPOLOGY_MISMATCH), "topology mismatch");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_PROTOCOL_ERROR), "protocol error");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_CHECKSUM_ERROR), "checksum error");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_TIMED_OUT), "timed out");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_UNAVAILABLE), "unavailable");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_CONNECTION_LOST), "connection lost");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_OUT_OF_RESOURCES), "out of resources");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_INVALID_HANDLE), "invalid handle");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_BUFFER_TOO_SMALL), "buffer too small");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_UNAUTHORIZED), "unauthorized");
  failures += expect_string(sagr_status_string(SAGR_STATUS_BUSY), "busy");
  failures += expect_string(
      sagr_status_string(SAGR_STATUS_CANCELLED), "cancelled");
  failures += expect_string(sagr_status_string(INT32_C(12345)), "unknown status");
  failures += expect_options_defaults();
  failures += expect_queue_option_defaults();
  failures += expect_memory_option_defaults();
  failures += expect_signal_option_defaults();
  failures += expect_dispatch_option_defaults();
  failures += expect_managed_option_defaults();
  failures += expect_exact_managed_option_validation();

  return failures == 0 ? 0 : 1;
}
