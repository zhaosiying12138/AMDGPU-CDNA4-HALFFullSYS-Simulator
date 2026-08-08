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

  if (abi_version != SAGR_ABI_VERSION ||
      strcmp(SAGR_VERSION_STRING, "0.4.0") != 0 ||
      SAGR_ABI_VERSION_DECODE_MAJOR(abi_version) != SAGR_ABI_VERSION_MAJOR ||
      SAGR_ABI_VERSION_DECODE_MINOR(abi_version) != SAGR_ABI_VERSION_MINOR ||
      SAGR_ABI_VERSION_MAJOR != 1 || SAGR_ABI_VERSION_MINOR != 3 ||
      SAGR_CAPABILITY_QUEUE_MASK != UINT64_C(2) ||
      SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST != UINT64_C(2) ||
      SAGR_CAPABILITY_MEMORY_MASK != UINT64_C(4) ||
      SAGR_MEMORY_MAX_TRANSFER_BYTES != UINT64_C(16777216)) {
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

  return failures == 0 ? 0 : 1;
}
