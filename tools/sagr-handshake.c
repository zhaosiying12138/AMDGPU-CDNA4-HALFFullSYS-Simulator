/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <self_amdgpu_runtime/runtime.h>

static void usage(FILE *stream, const char *program) {
  fprintf(stream,
          "usage: %s --endpoint PATH [--expected-daemon-uuid UUID] "
          "[--expected-job-uuid UUID --expected-rank N --expected-world N] "
          "[--expected-epoch N] [--min-version MAJOR.MINOR] "
          "[--max-version MAJOR.MINOR] [--offer-cap-bit N] "
          "[--require-cap-bit N] [--timeout-ms N|infinite] [--hold-ms N] "
          "[--queue-depth N --doorbells N --command-kind 0|1|2] "
          "[--memory-bytes N [--memory-alignment 4096|65536] "
          "[--memory-reuse]]\n",
          program);
}

typedef struct queue_cli_options {
  uint32_t enabled;
  uint32_t depth;
  uint32_t doorbells;
  uint64_t command_kind;
} queue_cli_options_t;

typedef struct queue_cli_result {
  sagr_queue_info_t info;
  uint64_t *sequences;
  uint32_t sequence_count;
  uint64_t command_kind;
  sagr_status_t completion_status;
  int32_t completion_wire_status;
  uint64_t completion_error_code;
} queue_cli_result_t;

typedef struct memory_cli_options {
  uint32_t enabled;
  uint32_t reuse;
  uint64_t bytes;
  uint64_t alignment;
} memory_cli_options_t;

typedef struct memory_cli_result {
  sagr_memory_info_t info;
  uint32_t initial_zero;
  uint32_t match;
  uint32_t freed;
  uint32_t reused;
  uint32_t reuse_zero;
  uint32_t pattern_crc;
  uint32_t returned_crc;
  uint64_t reuse_allocation_id;
  uint64_t reuse_generation;
  uint64_t reuse_simulated_va;
  uint32_t reuse_freed;
} memory_cli_result_t;

static int hex_value(char character) {
  if (character >= '0' && character <= '9') {
    return character - '0';
  }
  if (character >= 'a' && character <= 'f') {
    return character - 'a' + 10;
  }
  if (character >= 'A' && character <= 'F') {
    return character - 'A' + 10;
  }
  return -1;
}

static int parse_uuid(const char *text, uint8_t uuid[16]) {
  const size_t length = text == NULL ? 0 : strlen(text);
  size_t text_index;
  if (length != 32 && length != 36) {
    return -1;
  }
  uint32_t nibble_count = 0;
  memset(uuid, 0, 16);
  for (text_index = 0; text_index < length; ++text_index) {
    const int value = hex_value(text[text_index]);
    if (length == 36 &&
        (text_index == 8 || text_index == 13 || text_index == 18 ||
         text_index == 23)) {
      if (text[text_index] != '-') {
        return -1;
      }
      continue;
    }
    if (value < 0 || nibble_count >= 32) {
      return -1;
    }
    if ((nibble_count & 1U) == 0) {
      uuid[nibble_count / 2U] = (uint8_t)((uint32_t)value << 4);
    } else {
      uuid[nibble_count / 2U] =
          (uint8_t)(uuid[nibble_count / 2U] | (uint8_t)value);
    }
    ++nibble_count;
  }
  return nibble_count == 32 ? 0 : -1;
}

static int parse_u64(const char *text, uint64_t *value) {
  char *end = NULL;
  unsigned long long parsed;
  const unsigned char *cursor;
  if (text == NULL || text[0] == '\0' || text[0] == '-' || text[0] == '+') {
    return -1;
  }
  for (cursor = (const unsigned char *)text; *cursor != 0; ++cursor) {
    if (isspace(*cursor) != 0) {
      return -1;
    }
  }
  errno = 0;
  parsed = strtoull(text, &end, 0);
  if (errno != 0 || end == text || *end != '\0') {
    return -1;
  }
  *value = (uint64_t)parsed;
  return 0;
}

static int parse_u32(const char *text, uint32_t *value) {
  uint64_t parsed;
  if (parse_u64(text, &parsed) != 0 || parsed > UINT32_MAX) {
    return -1;
  }
  *value = (uint32_t)parsed;
  return 0;
}

static int parse_version(const char *text, uint16_t *major, uint16_t *minor) {
  char *separator = NULL;
  char *end = NULL;
  unsigned long parsed_major;
  unsigned long parsed_minor;
  const unsigned char *cursor;
  uint32_t separators = 0;
  if (text == NULL || text[0] == '\0' || text[0] == '-' || text[0] == '+') {
    return -1;
  }
  for (cursor = (const unsigned char *)text; *cursor != 0; ++cursor) {
    if (*cursor == '.') {
      ++separators;
    } else if (isdigit(*cursor) == 0) {
      return -1;
    }
  }
  if (separators != 1) {
    return -1;
  }
  errno = 0;
  parsed_major = strtoul(text, &separator, 10);
  if (errno != 0 || separator == text || *separator != '.') {
    return -1;
  }
  if (separator[1] == '\0' || separator[1] == '-' || separator[1] == '+') {
    return -1;
  }
  errno = 0;
  parsed_minor = strtoul(separator + 1, &end, 10);
  if (errno != 0 || end == separator + 1 || *end != '\0' ||
      parsed_major > UINT16_MAX || parsed_minor > UINT16_MAX) {
    return -1;
  }
  *major = (uint16_t)parsed_major;
  *minor = (uint16_t)parsed_minor;
  return 0;
}

static int hold_connection(uint64_t milliseconds) {
  struct timespec deadline;
  const uint64_t seconds = milliseconds / UINT64_C(1000);
  uint64_t nanoseconds =
      (milliseconds % UINT64_C(1000)) * UINT64_C(1000000);
  uint64_t target_seconds;
  time_t converted;
  int result;
  if (milliseconds == 0) {
    return 0;
  }
  if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0 || deadline.tv_sec < 0 ||
      seconds > UINT64_MAX - (uint64_t)deadline.tv_sec) {
    errno = EOVERFLOW;
    return -1;
  }
  target_seconds = (uint64_t)deadline.tv_sec + seconds;
  nanoseconds += (uint64_t)deadline.tv_nsec;
  if (nanoseconds >= UINT64_C(1000000000)) {
    if (target_seconds == UINT64_MAX) {
      errno = EOVERFLOW;
      return -1;
    }
    ++target_seconds;
    nanoseconds -= UINT64_C(1000000000);
  }
  converted = (time_t)target_seconds;
  if ((uint64_t)converted != target_seconds) {
    errno = EOVERFLOW;
    return -1;
  }
  deadline.tv_sec = converted;
  deadline.tv_nsec = (long)nanoseconds;
  do {
    result = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, NULL);
  } while (result == EINTR);
  if (result != 0) {
    errno = result;
    return -1;
  }
  return 0;
}

static void print_json_string(FILE *stream, const char *text) {
  const unsigned char *cursor = (const unsigned char *)text;
  fputc('"', stream);
  while (*cursor != 0) {
    if (*cursor == '"' || *cursor == '\\') {
      fputc('\\', stream);
      fputc(*cursor, stream);
    } else if (*cursor < 0x20U) {
      fprintf(stream, "\\u%04x", (unsigned int)*cursor);
    } else {
      fputc(*cursor, stream);
    }
    ++cursor;
  }
  fputc('"', stream);
}

static void print_uuid(const uint8_t uuid[16]) {
  uint32_t index;
  putchar('"');
  for (index = 0; index < 16; ++index) {
    printf("%02x", (unsigned int)uuid[index]);
  }
  putchar('"');
}

static int parse_arguments(int argc, char **argv, const char **endpoint,
                           sagr_instance_open_options_t *options,
                           uint64_t *hold_ms,
                           queue_cli_options_t *queue_options,
                           memory_cli_options_t *memory_options) {
  int index;
  int have_job = 0;
  int have_rank = 0;
  int have_world = 0;
  int have_queue_depth = 0;
  int have_doorbells = 0;
  int have_command_kind = 0;
  int have_memory_bytes = 0;
  int have_memory_alignment = 0;
  memory_options->alignment = SAGR_MEMORY_ALIGNMENT_4K;
  for (index = 1; index < argc; ++index) {
    const char *argument = argv[index];
    if (strcmp(argument, "--help") == 0) {
      usage(stdout, argv[0]);
      exit(0);
    }
    if (strcmp(argument, "--memory-reuse") == 0) {
      memory_options->reuse = 1;
      continue;
    }
    if (index + 1 >= argc) {
      return -1;
    }
    if (strcmp(argument, "--endpoint") == 0) {
      *endpoint = argv[++index];
    } else if (strcmp(argument, "--expected-daemon-uuid") == 0 ||
               strcmp(argument, "--expected-uuid") == 0) {
      if (parse_uuid(argv[++index], options->expected_daemon_uuid) != 0) {
        return -1;
      }
    } else if (strcmp(argument, "--expected-job-uuid") == 0) {
      if (parse_uuid(argv[++index], options->expected_job_uuid) != 0) {
        return -1;
      }
      have_job = 1;
    } else if (strcmp(argument, "--expected-epoch") == 0) {
      if (parse_u64(argv[++index], &options->expected_epoch) != 0) {
        return -1;
      }
    } else if (strcmp(argument, "--expected-rank") == 0) {
      if (parse_u32(argv[++index], &options->expected_rank) != 0) {
        return -1;
      }
      have_rank = 1;
    } else if (strcmp(argument, "--expected-world") == 0 ||
               strcmp(argument, "--expected-world-size") == 0) {
      if (parse_u32(argv[++index], &options->expected_world_size) != 0) {
        return -1;
      }
      have_world = 1;
    } else if (strcmp(argument, "--min-version") == 0) {
      if (parse_version(argv[++index], &options->minimum_version_major,
                        &options->minimum_version_minor) != 0) {
        return -1;
      }
    } else if (strcmp(argument, "--max-version") == 0) {
      if (parse_version(argv[++index], &options->maximum_version_major,
                        &options->maximum_version_minor) != 0) {
        return -1;
      }
    } else if (strcmp(argument, "--offer-cap-bit") == 0) {
      uint32_t bit;
      if (parse_u32(argv[++index], &bit) != 0 || bit >= 256) {
        return -1;
      }
      options->offered_capabilities[bit / 64U] |=
          UINT64_C(1) << (bit % 64U);
    } else if (strcmp(argument, "--require-cap-bit") == 0) {
      uint32_t bit;
      if (parse_u32(argv[++index], &bit) != 0 || bit >= 256) {
        return -1;
      }
      options->required_capabilities[bit / 64U] |=
          UINT64_C(1) << (bit % 64U);
    } else if (strcmp(argument, "--timeout-ms") == 0) {
      uint64_t milliseconds;
      const char *value = argv[++index];
      if (strcmp(value, "infinite") == 0) {
        options->open_timeout_ns = UINT64_MAX;
      } else if (parse_u64(value, &milliseconds) != 0 ||
                 milliseconds > UINT64_MAX / UINT64_C(1000000)) {
        return -1;
      } else {
        options->open_timeout_ns = milliseconds * UINT64_C(1000000);
      }
    } else if (strcmp(argument, "--hold-ms") == 0) {
      if (parse_u64(argv[++index], hold_ms) != 0) {
        return -1;
      }
    } else if (strcmp(argument, "--queue-depth") == 0) {
      if (parse_u32(argv[++index], &queue_options->depth) != 0 ||
          queue_options->depth == 0 ||
          queue_options->depth > SAGR_QUEUE_MAX_DEPTH) {
        return -1;
      }
      have_queue_depth = 1;
    } else if (strcmp(argument, "--doorbells") == 0) {
      if (parse_u32(argv[++index], &queue_options->doorbells) != 0 ||
          queue_options->doorbells > UINT32_C(65536)) {
        return -1;
      }
      have_doorbells = 1;
    } else if (strcmp(argument, "--command-kind") == 0) {
      if (parse_u64(argv[++index], &queue_options->command_kind) != 0 ||
          (queue_options->command_kind != SAGR_QUEUE_COMMAND_NOOP &&
           queue_options->command_kind !=
               SAGR_QUEUE_COMMAND_CONTROL_TEST &&
           queue_options->command_kind !=
               SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST)) {
        return -1;
      }
      have_command_kind = 1;
    } else if (strcmp(argument, "--memory-bytes") == 0) {
      if (parse_u64(argv[++index], &memory_options->bytes) != 0 ||
          memory_options->bytes == 0 ||
          memory_options->bytes > SAGR_MEMORY_MAX_TRANSFER_BYTES) {
        return -1;
      }
      have_memory_bytes = 1;
    } else if (strcmp(argument, "--memory-alignment") == 0) {
      if (parse_u64(argv[++index], &memory_options->alignment) != 0 ||
          (memory_options->alignment != SAGR_MEMORY_ALIGNMENT_4K &&
           memory_options->alignment != SAGR_MEMORY_ALIGNMENT_64K)) {
        return -1;
      }
      have_memory_alignment = 1;
    } else {
      return -1;
    }
  }
  if (*endpoint == NULL || have_job != have_rank || have_job != have_world) {
    return -1;
  }
  if (have_job != 0 &&
      (options->expected_world_size == 0 ||
       options->expected_rank >= options->expected_world_size)) {
    return -1;
  }
  if (have_queue_depth != have_doorbells ||
      have_queue_depth != have_command_kind) {
    return -1;
  }
  if (have_queue_depth != 0) {
    queue_options->enabled = 1;
    options->offered_capabilities[SAGR_CAPABILITY_QUEUE_WORD] |=
        SAGR_CAPABILITY_QUEUE_MASK;
    options->required_capabilities[SAGR_CAPABILITY_QUEUE_WORD] |=
        SAGR_CAPABILITY_QUEUE_MASK;
  }
  if (have_memory_alignment != 0 && have_memory_bytes == 0) {
    return -1;
  }
  if (memory_options->reuse != 0 && have_memory_bytes == 0) {
    return -1;
  }
  if (have_memory_bytes != 0) {
    memory_options->enabled = 1;
    options->offered_capabilities[SAGR_CAPABILITY_MEMORY_WORD] |=
        SAGR_CAPABILITY_MEMORY_MASK;
    options->required_capabilities[SAGR_CAPABILITY_MEMORY_WORD] |=
        SAGR_CAPABILITY_MEMORY_MASK;
  }
  return 0;
}

static void set_local_error(sagr_error_info_t *error, sagr_status_t status,
                            int native_errno, const char *message) {
  memset(error, 0, sizeof(*error));
  error->struct_size = (uint32_t)sizeof(*error);
  error->status = status;
  error->wire_status = -1;
  error->native_errno = native_errno;
  (void)snprintf(error->message, sizeof(error->message), "%s", message);
}

static uint32_t cli_crc32c(const uint8_t *data, size_t size) {
  uint32_t crc = UINT32_MAX;
  size_t index;
  for (index = 0; index < size; ++index) {
    uint32_t bit;
    crc ^= data[index];
    for (bit = 0; bit < 8; ++bit) {
      const uint32_t mask = UINT32_C(0) - (crc & UINT32_C(1));
      crc = (crc >> 1) ^ (UINT32_C(0x82f63b78) & mask);
    }
  }
  return ~crc;
}

static int memory_bytes_are_zero(const uint8_t *bytes, size_t size) {
  uint8_t combined = 0;
  size_t index;
  for (index = 0; index < size; ++index) {
    combined = (uint8_t)(combined | bytes[index]);
  }
  return combined == 0;
}

static void fill_memory_pattern(uint8_t *bytes, size_t size) {
  size_t index;
  for (index = 0; index < size; ++index) {
    const uint64_t position = (uint64_t)index;
    bytes[index] = (uint8_t)((position * UINT64_C(131) +
                              (position >> 8) + UINT64_C(17)) &
                             UINT64_C(0xff));
  }
}

static sagr_status_t run_memory_roundtrip(
    sagr_instance_t instance, const sagr_instance_open_options_t *open_options,
    const memory_cli_options_t *memory_options, memory_cli_result_t *result,
    sagr_error_info_t *error, const char **phase) {
  sagr_memory_allocate_options_t allocate_options;
  sagr_memory_operation_options_t operation_options;
  sagr_memory_info_t reuse_info;
  sagr_memory_t memory = NULL;
  sagr_memory_t reuse_memory = NULL;
  uint8_t *pattern = NULL;
  uint8_t *returned = NULL;
  const size_t byte_count = (size_t)memory_options->bytes;
  sagr_status_t status;

  memset(result, 0, sizeof(*result));
  memset(&reuse_info, 0, sizeof(reuse_info));
  status = sagr_memory_allocate_options_init(
      &allocate_options, (uint32_t)sizeof(allocate_options));
  if (status == SAGR_STATUS_SUCCESS) {
    status = sagr_memory_operation_options_init(
        &operation_options, (uint32_t)sizeof(operation_options));
  }
  if (status != SAGR_STATUS_SUCCESS) {
    set_local_error(error, status, 0, "could not initialize memory options");
    *phase = "memory_options";
    return status;
  }
  allocate_options.size_bytes = memory_options->bytes;
  allocate_options.alignment_bytes = memory_options->alignment;
  operation_options.timeout_ns = open_options->open_timeout_ns;
  pattern = (uint8_t *)malloc(byte_count);
  returned = (uint8_t *)malloc(byte_count);
  if (pattern == NULL || returned == NULL) {
    set_local_error(error, SAGR_STATUS_OUT_OF_RESOURCES, errno,
                    "could not allocate memory roundtrip buffers");
    *phase = "memory_host_allocate";
    status = SAGR_STATUS_OUT_OF_RESOURCES;
    goto done;
  }

  *phase = "memory_allocate";
  status = sagr_memory_allocate(
      instance, &allocate_options, &operation_options, &memory, &result->info,
      (uint32_t)sizeof(result->info), error, (uint32_t)sizeof(*error));
  if (status != SAGR_STATUS_SUCCESS) {
    goto done;
  }

  memset(returned, 0xa5, byte_count);
  *phase = "memory_initial_d2h";
  status = sagr_memory_copy_to_host(memory, 0, returned,
                                    memory_options->bytes, &operation_options,
                                    error, (uint32_t)sizeof(*error));
  if (status != SAGR_STATUS_SUCCESS) {
    goto done;
  }
  result->initial_zero =
      (uint32_t)memory_bytes_are_zero(returned, byte_count);
  if (result->initial_zero == 0) {
    status = SAGR_STATUS_CHECKSUM_ERROR;
    set_local_error(error, status, 0,
                    "new simulated allocation was not zero initialized");
    *phase = "memory_initial_zero";
    goto done;
  }

  fill_memory_pattern(pattern, byte_count);
  result->pattern_crc = cli_crc32c(pattern, byte_count);
  *phase = "memory_h2d";
  status = sagr_memory_copy_from_host(
      memory, 0, pattern, memory_options->bytes, &operation_options, error,
      (uint32_t)sizeof(*error));
  if (status != SAGR_STATUS_SUCCESS) {
    goto done;
  }

  memset(returned, 0xa5, byte_count);
  *phase = "memory_d2h";
  status = sagr_memory_copy_to_host(memory, 0, returned,
                                    memory_options->bytes, &operation_options,
                                    error, (uint32_t)sizeof(*error));
  if (status != SAGR_STATUS_SUCCESS) {
    goto done;
  }
  result->returned_crc = cli_crc32c(returned, byte_count);
  result->match =
      (uint32_t)(memcmp(pattern, returned, byte_count) == 0 &&
                 result->pattern_crc == result->returned_crc);
  if (result->match == 0) {
    status = SAGR_STATUS_CHECKSUM_ERROR;
    set_local_error(error, status, 0,
                    "simulated memory roundtrip bytes did not match");
    *phase = "memory_compare";
    goto done;
  }

  *phase = "memory_free";
  status = sagr_memory_free(&memory, &operation_options, error,
                            (uint32_t)sizeof(*error));
  if (status != SAGR_STATUS_SUCCESS) {
    goto done;
  }
  result->freed = 1;

  if (memory_options->reuse != 0) {
    *phase = "memory_reuse_allocate";
    status = sagr_memory_allocate(
        instance, &allocate_options, &operation_options, &reuse_memory,
        &reuse_info, (uint32_t)sizeof(reuse_info), error,
        (uint32_t)sizeof(*error));
    if (status != SAGR_STATUS_SUCCESS) {
      goto done;
    }
    result->reused = 1;
    result->reuse_allocation_id = reuse_info.allocation_id;
    result->reuse_generation = reuse_info.generation;
    result->reuse_simulated_va = reuse_info.simulated_va;
    if (reuse_info.allocation_id != result->info.allocation_id ||
        reuse_info.simulated_va != result->info.simulated_va ||
        reuse_info.generation == 0 ||
        reuse_info.generation == result->info.generation) {
      status = SAGR_STATUS_PROTOCOL_ERROR;
      set_local_error(error, status, 0,
                      "reallocated memory did not reuse the slot canonically");
      *phase = "memory_reuse_identity";
      goto done;
    }
    memset(returned, 0xa5, byte_count);
    *phase = "memory_reuse_d2h";
    status = sagr_memory_copy_to_host(
        reuse_memory, 0, returned, memory_options->bytes, &operation_options,
        error, (uint32_t)sizeof(*error));
    if (status != SAGR_STATUS_SUCCESS) {
      goto done;
    }
    result->reuse_zero =
        (uint32_t)memory_bytes_are_zero(returned, byte_count);
    if (result->reuse_zero == 0) {
      status = SAGR_STATUS_CHECKSUM_ERROR;
      set_local_error(error, status, 0,
                      "reused simulated allocation was not zero initialized");
      *phase = "memory_reuse_zero";
      goto done;
    }
    *phase = "memory_reuse_free";
    status = sagr_memory_free(&reuse_memory, &operation_options, error,
                              (uint32_t)sizeof(*error));
    if (status != SAGR_STATUS_SUCCESS) {
      goto done;
    }
    result->reuse_freed = 1;
  }

done:
  free(returned);
  free(pattern);
  return status;
}

static sagr_status_t run_queue_control(
    sagr_instance_t instance, const sagr_instance_open_options_t *open_options,
    const queue_cli_options_t *queue_options, queue_cli_result_t *result,
    sagr_error_info_t *error, const char **phase) {
  sagr_queue_create_options_t create_options;
  sagr_queue_operation_options_t operation_options;
  sagr_queue_completion_t completion;
  sagr_queue_t queue = NULL;
  sagr_status_t status;
  uint32_t index;
  memset(result, 0, sizeof(*result));
  result->sequence_count = queue_options->doorbells;
  result->command_kind = queue_options->command_kind;
  if (queue_options->doorbells != 0) {
    result->sequences = (uint64_t *)calloc(queue_options->doorbells,
                                           sizeof(result->sequences[0]));
    if (result->sequences == NULL) {
      set_local_error(error, SAGR_STATUS_OUT_OF_RESOURCES, errno,
                      "could not allocate queue sequence output");
      *phase = "queue_allocate";
      return SAGR_STATUS_OUT_OF_RESOURCES;
    }
  }
  status = sagr_queue_create_options_init(&create_options,
                                          (uint32_t)sizeof(create_options));
  if (status == SAGR_STATUS_SUCCESS) {
    status = sagr_queue_operation_options_init(
        &operation_options, (uint32_t)sizeof(operation_options));
  }
  if (status != SAGR_STATUS_SUCCESS) {
    set_local_error(error, status, 0, "could not initialize queue options");
    *phase = "queue_options";
    return status;
  }
  create_options.depth = queue_options->depth;
  operation_options.timeout_ns = open_options->open_timeout_ns;
  *phase = "queue_create";
  status = sagr_queue_create(
      instance, &create_options, &operation_options, &queue, &result->info,
      (uint32_t)sizeof(result->info), error, (uint32_t)sizeof(*error));
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  for (index = 0; index < queue_options->doorbells; ++index) {
    *phase = "queue_ring";
    status = sagr_queue_ring_doorbell(
        queue, queue_options->command_kind, &operation_options,
        &result->sequences[index], error, (uint32_t)sizeof(*error));
    if (status != SAGR_STATUS_SUCCESS) {
      return status;
    }
    *phase = "queue_wait";
    status = sagr_queue_wait(queue, result->sequences[index],
                             &operation_options, &completion,
                             (uint32_t)sizeof(completion), error,
                             (uint32_t)sizeof(*error));
    if (status == SAGR_STATUS_SUCCESS) {
      result->completion_status = completion.status;
      result->completion_wire_status = completion.wire_status;
      result->completion_error_code = completion.error_code;
    } else if (queue_options->command_kind ==
                   SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST &&
               status == SAGR_STATUS_INTERNAL_ERROR &&
               completion.status == SAGR_STATUS_INTERNAL_ERROR &&
               completion.wire_status == INT32_C(10) &&
               completion.value == SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST &&
               completion.error_code == UINT64_C(1)) {
      result->completion_status = completion.status;
      result->completion_wire_status = completion.wire_status;
      result->completion_error_code = completion.error_code;
    } else {
      return status;
    }
  }
  *phase = "queue_destroy";
  return sagr_queue_destroy(&queue, &operation_options, error,
                            (uint32_t)sizeof(*error));
}

static void print_failure_json(const char *phase, sagr_status_t status,
                               const sagr_error_info_t *error) {
  fputs("{\"status\":", stderr);
  fprintf(stderr, "%" PRId32 ",\"status_name\":", status);
  print_json_string(stderr, sagr_status_string(status));
  fputs(",\"phase\":", stderr);
  print_json_string(stderr, phase);
  fprintf(stderr, ",\"wire_status\":%" PRId32
                  ",\"native_errno\":%" PRId32 ",\"message\":",
          error->wire_status, error->native_errno);
  print_json_string(stderr, error->message);
  fputs("}\n", stderr);
}

int main(int argc, char **argv) {
  const char *endpoint = NULL;
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  sagr_instance_info_t info;
  sagr_instance_t instance = NULL;
  sagr_status_t status;
  queue_cli_options_t queue_options;
  queue_cli_result_t queue_result;
  memory_cli_options_t memory_options;
  memory_cli_result_t memory_result;
  const char *failure_phase = "handshake";
  uint32_t index;
  uint64_t hold_ms = 0;

  memset(&queue_options, 0, sizeof(queue_options));
  memset(&queue_result, 0, sizeof(queue_result));
  memset(&memory_options, 0, sizeof(memory_options));
  memset(&memory_result, 0, sizeof(memory_result));
  status = sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  if (status != SAGR_STATUS_SUCCESS ||
      parse_arguments(argc, argv, &endpoint, &options, &hold_ms,
                      &queue_options, &memory_options) != 0) {
    usage(stderr, argv[0]);
    return 2;
  }
  status = sagr_instance_open(endpoint, &options, &instance, &error,
                              (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS) {
    print_failure_json(failure_phase, status, &error);
    return 1;
  }
  status = sagr_instance_get_info(instance, &info, (uint32_t)sizeof(info));
  if (status != SAGR_STATUS_SUCCESS) {
    (void)sagr_instance_close(&instance);
    fprintf(stderr, "{\"status\":%" PRId32 "}\n", status);
    return 1;
  }
  if (queue_options.enabled != 0) {
    status = run_queue_control(instance, &options, &queue_options,
                               &queue_result, &error, &failure_phase);
    if (status != SAGR_STATUS_SUCCESS) {
      (void)sagr_instance_close(&instance);
      print_failure_json(failure_phase, status, &error);
      free(queue_result.sequences);
      return 1;
    }
  }
  if (memory_options.enabled != 0) {
    status = run_memory_roundtrip(instance, &options, &memory_options,
                                  &memory_result, &error, &failure_phase);
    if (status != SAGR_STATUS_SUCCESS) {
      (void)sagr_instance_close(&instance);
      print_failure_json(failure_phase, status, &error);
      free(queue_result.sequences);
      return 1;
    }
  }

  printf("{\"status\":0,\"selected_version\":\"%" PRIu16 ".%" PRIu16
         "\",\"capability_words\":[",
         info.selected_version_major, info.selected_version_minor);
  for (index = 0; index < SAGR_CAPABILITY_WORD_COUNT; ++index) {
    printf("%s\"0x%016" PRIx64 "\"", index == 0 ? "" : ",",
           info.negotiated_capabilities[index]);
  }
  fputs("],\"daemon_uuid\":", stdout);
  print_uuid(info.daemon_uuid);
  fputs(",\"job_uuid\":", stdout);
  print_uuid(info.job_uuid);
  printf(",\"connection_id\":\"0x%016" PRIx64
         "\",\"epoch\":\"0x%016" PRIx64
         "\",\"rank\":%" PRIu32 ",\"world_size\":%" PRIu32
         ",\"maximum_record_bytes\":%" PRIu32
         ",\"request_id\":\"0x%016" PRIx64
         "\",\"peer_uid\":%" PRIu32 ",\"peer_pid\":%" PRIu32,
         info.connection_id, info.epoch, info.rank, info.world_size,
         info.maximum_record_bytes, info.request_id, info.peer_uid,
         info.peer_pid);
  if (queue_options.enabled != 0) {
    printf(",\"queue\":{\"status\":0,\"queue_id\":\"0x%016" PRIx64
           "\",\"generation\":\"0x%016" PRIx64
           "\",\"depth\":%" PRIu32 ",\"command_kind\":%" PRIu64
           ",\"completion_status\":%" PRId32
           ",\"completion_wire_status\":%" PRId32
           ",\"completion_error_code\":%" PRIu64 ",\"sequences\":[",
           queue_result.info.queue_id, queue_result.info.generation,
           queue_result.info.depth, queue_result.command_kind,
           queue_result.completion_status,
           queue_result.completion_wire_status,
           queue_result.completion_error_code);
    for (index = 0; index < queue_result.sequence_count; ++index) {
      printf("%s\"0x%016" PRIx64 "\"", index == 0 ? "" : ",",
             queue_result.sequences[index]);
    }
    fputs("]}", stdout);
  }
  if (memory_options.enabled != 0) {
    printf(",\"memory\":{\"status\":0,\"allocation_id\":\"0x%016" PRIx64
           "\",\"generation\":\"0x%016" PRIx64
           "\",\"simulated_va\":\"0x%016" PRIx64
           "\",\"size_bytes\":%" PRIu64
           ",\"alignment_bytes\":%" PRIu64
           ",\"initial_zero\":%s,\"pattern_crc32c\":\"0x%08" PRIx32
           "\",\"returned_crc32c\":\"0x%08" PRIx32
           "\",\"match\":%s,\"freed\":%s",
           memory_result.info.allocation_id, memory_result.info.generation,
           memory_result.info.simulated_va, memory_result.info.size_bytes,
           memory_result.info.alignment_bytes,
           memory_result.initial_zero != 0 ? "true" : "false",
           memory_result.pattern_crc, memory_result.returned_crc,
           memory_result.match != 0 ? "true" : "false",
           memory_result.freed != 0 ? "true" : "false");
    if (memory_options.reuse != 0) {
      printf(",\"reuse\":{\"allocation_id\":\"0x%016" PRIx64
             "\",\"generation\":\"0x%016" PRIx64
             "\",\"simulated_va\":\"0x%016" PRIx64
             "\",\"initial_zero\":%s,\"freed\":%s}",
             memory_result.reuse_allocation_id,
             memory_result.reuse_generation,
             memory_result.reuse_simulated_va,
             memory_result.reuse_zero != 0 ? "true" : "false",
             memory_result.reuse_freed != 0 ? "true" : "false");
    }
    fputc('}', stdout);
  }
  fputs("}\n", stdout);
  if (fflush(stdout) != 0 || hold_connection(hold_ms) != 0) {
    const int native_error = errno;
    (void)sagr_instance_close(&instance);
    free(queue_result.sequences);
    fprintf(stderr,
            "{\"status\":%d,\"status_name\":\"internal error\","
            "\"wire_status\":-1,\"native_errno\":%d,"
            "\"message\":\"could not hold established connection\"}\n",
            SAGR_STATUS_INTERNAL_ERROR, native_error);
    return 1;
  }
  (void)sagr_instance_close(&instance);
  free(queue_result.sequences);
  return 0;
}
