/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_RUNTIME_H
#define SELF_AMDGPU_RUNTIME_RUNTIME_H

#include <stdint.h>

#include <self_amdgpu_runtime/export.h>
#include <self_amdgpu_runtime/version.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t sagr_status_t;

enum {
  SAGR_STATUS_SUCCESS = 0,
  SAGR_STATUS_INVALID_ARGUMENT = 1,
  SAGR_STATUS_NOT_SUPPORTED = 2,
  SAGR_STATUS_INTERNAL_ERROR = 3,
  SAGR_STATUS_VERSION_MISMATCH = 4,
  SAGR_STATUS_CAPABILITY_MISMATCH = 5,
  SAGR_STATUS_ENDPOINT_NOT_FOUND = 6,
  SAGR_STATUS_INSTANCE_MISMATCH = 7,
  SAGR_STATUS_TOPOLOGY_MISMATCH = 8,
  SAGR_STATUS_PROTOCOL_ERROR = 9,
  SAGR_STATUS_CHECKSUM_ERROR = 10,
  SAGR_STATUS_TIMED_OUT = 11,
  SAGR_STATUS_UNAVAILABLE = 12,
  SAGR_STATUS_CONNECTION_LOST = 13,
  SAGR_STATUS_OUT_OF_RESOURCES = 14,
  SAGR_STATUS_INVALID_HANDLE = 15,
  SAGR_STATUS_BUFFER_TOO_SMALL = 16,
  SAGR_STATUS_UNAUTHORIZED = 17,
  SAGR_STATUS_BUSY = 18,
  SAGR_STATUS_CANCELLED = 19
};

#define SAGR_UUID_SIZE UINT32_C(16)
#define SAGR_CAPABILITY_WORD_COUNT UINT32_C(4)
#define SAGR_ERROR_MESSAGE_SIZE UINT32_C(128)
#define SAGR_DEFAULT_OPEN_TIMEOUT_NS UINT64_C(5000000000)
#define SAGR_INSTANCE_RANK_WILDCARD UINT32_MAX

#define SAGR_CAPABILITY_TOPOLOGY_WORD UINT32_C(0)
#define SAGR_CAPABILITY_TOPOLOGY_MASK UINT64_C(1)

typedef struct sagr_instance *sagr_instance_t;

/*
 * All public structures contain fixed-width fields only. Callers initialize
 * options with sagr_instance_open_options_init() and pass the actual allocation
 * size. Reserved fields and flags must remain zero.
 */
typedef struct sagr_instance_open_options {
  uint32_t struct_size;
  uint32_t flags;
  uint16_t minimum_version_major;
  uint16_t minimum_version_minor;
  uint16_t maximum_version_major;
  uint16_t maximum_version_minor;
  uint64_t open_timeout_ns;
  uint64_t offered_capabilities[SAGR_CAPABILITY_WORD_COUNT];
  uint64_t required_capabilities[SAGR_CAPABILITY_WORD_COUNT];
  uint8_t expected_daemon_uuid[SAGR_UUID_SIZE];
  uint8_t expected_job_uuid[SAGR_UUID_SIZE];
  uint64_t expected_epoch;
  uint32_t expected_rank;
  uint32_t expected_world_size;
  /*
   * When nonzero, absolute_deadline_ns is the CLOCK_MONOTONIC deadline and
   * takes precedence over open_timeout_ns. cancel_fd is -1 when disabled;
   * otherwise readability, hangup, or error cancels the open. The caller owns
   * cancel_fd, must set FD_CLOEXEC, and must keep it open until
   * sagr_instance_open() returns. An expired deadline takes precedence over
   * simultaneous cancellation readiness.
   */
  uint64_t absolute_deadline_ns;
  int32_t cancel_fd;
  uint32_t reserved0;
  uint8_t reserved[8];
} sagr_instance_open_options_t;

typedef struct sagr_instance_info {
  uint32_t struct_size;
  uint32_t flags;
  uint16_t selected_version_major;
  uint16_t selected_version_minor;
  uint32_t maximum_record_bytes;
  uint64_t negotiated_capabilities[SAGR_CAPABILITY_WORD_COUNT];
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t job_uuid[SAGR_UUID_SIZE];
  uint64_t connection_id;
  uint64_t epoch;
  uint32_t rank;
  uint32_t world_size;
  uint32_t peer_uid;
  uint32_t peer_pid;
  uint64_t request_id;
  uint8_t reserved[32];
} sagr_instance_info_t;

typedef struct sagr_error_info {
  uint32_t struct_size;
  sagr_status_t status;
  int32_t wire_status;
  int32_t native_errno;
  char message[SAGR_ERROR_MESSAGE_SIZE];
  uint8_t reserved[16];
} sagr_error_info_t;

SAGR_API uint32_t sagr_abi_version(void);
SAGR_API const char *sagr_version_string(void);
SAGR_API const char *sagr_status_string(sagr_status_t status);

/* Set protocol 1.0, a five-second relative deadline, and no cancellation FD. */
SAGR_API sagr_status_t sagr_instance_open_options_init(
    sagr_instance_open_options_t *options, uint32_t options_size);

/*
 * Open performs exactly one AF_UNIX SOCK_SEQPACKET handshake attempt. When no
 * absolute deadline is supplied, a zero relative timeout selects the
 * five-second default and UINT64_MAX means no deadline. On every failure, a
 * non-null out_instance is set to NULL. native_errno is diagnostic only; the
 * returned sagr_status_t is authoritative.
 */
SAGR_API sagr_status_t sagr_instance_open(
    const char *endpoint, const sagr_instance_open_options_t *options,
    sagr_instance_t *out_instance, sagr_error_info_t *out_error,
    uint32_t error_size);
SAGR_API sagr_status_t sagr_instance_get_info(
    sagr_instance_t instance, sagr_instance_info_t *info,
    uint32_t info_size);
/* Local-only, idempotent for *instance == NULL, and always clears on success. */
SAGR_API sagr_status_t sagr_instance_close(sagr_instance_t *instance);

#ifdef __cplusplus
}
#endif

#endif
