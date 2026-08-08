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
          "[--require-cap-bit N] [--timeout-ms N|infinite] [--hold-ms N]\n",
          program);
}

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
                           uint64_t *hold_ms) {
  int index;
  int have_job = 0;
  int have_rank = 0;
  int have_world = 0;
  for (index = 1; index < argc; ++index) {
    const char *argument = argv[index];
    if (strcmp(argument, "--help") == 0) {
      usage(stdout, argv[0]);
      exit(0);
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
  return 0;
}

int main(int argc, char **argv) {
  const char *endpoint = NULL;
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  sagr_instance_info_t info;
  sagr_instance_t instance = NULL;
  sagr_status_t status;
  uint32_t index;
  uint64_t hold_ms = 0;

  status = sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  if (status != SAGR_STATUS_SUCCESS ||
      parse_arguments(argc, argv, &endpoint, &options, &hold_ms) != 0) {
    usage(stderr, argv[0]);
    return 2;
  }
  status = sagr_instance_open(endpoint, &options, &instance, &error,
                              (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS) {
    fputs("{\"status\":", stderr);
    fprintf(stderr, "%" PRId32 ",\"status_name\":", status);
    print_json_string(stderr, sagr_status_string(status));
    fprintf(stderr, ",\"wire_status\":%" PRId32
                    ",\"native_errno\":%" PRId32 ",\"message\":",
            error.wire_status, error.native_errno);
    print_json_string(stderr, error.message);
    fputs("}\n", stderr);
    return 1;
  }
  status = sagr_instance_get_info(instance, &info, (uint32_t)sizeof(info));
  if (status != SAGR_STATUS_SUCCESS) {
    (void)sagr_instance_close(&instance);
    fprintf(stderr, "{\"status\":%" PRId32 "}\n", status);
    return 1;
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
         "\",\"peer_uid\":%" PRIu32 ",\"peer_pid\":%" PRIu32 "}\n",
         info.connection_id, info.epoch, info.rank, info.world_size,
         info.maximum_record_bytes, info.request_id, info.peer_uid,
         info.peer_pid);
  if (fflush(stdout) != 0 || hold_connection(hold_ms) != 0) {
    const int native_error = errno;
    (void)sagr_instance_close(&instance);
    fprintf(stderr,
            "{\"status\":%d,\"status_name\":\"internal error\","
            "\"wire_status\":-1,\"native_errno\":%d,"
            "\"message\":\"could not hold established connection\"}\n",
            SAGR_STATUS_INTERNAL_ERROR, native_error);
    return 1;
  }
  (void)sagr_instance_close(&instance);
  return 0;
}
