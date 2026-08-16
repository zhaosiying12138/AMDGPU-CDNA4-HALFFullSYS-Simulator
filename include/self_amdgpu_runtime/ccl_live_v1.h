/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_CCL_LIVE_V1_H
#define SELF_AMDGPU_RUNTIME_CCL_LIVE_V1_H

#include <stdint.h>

#include <self_amdgpu_runtime/ccl_v1.h>
#include <self_amdgpu_runtime/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Host-only authenticated rendezvous for CCL v1.
 *
 * This layer does not construct collective descriptors, plan collectives, move
 * payloads, or own carrier sessions.  It turns one pre-fork capability socket
 * per rank into an exact rank-indexed table of connected Unix seqpacket FDs.
 */

#define SAGR_CCL_LIVE_V1_NO_RANK SAGR_CCL_V1_NO_RANK

typedef enum sagr_ccl_live_v1_phase {
  SAGR_CCL_LIVE_V1_PHASE_UNINITIALIZED = 0,
  SAGR_CCL_LIVE_V1_PHASE_CONFIGURING = 1,
  SAGR_CCL_LIVE_V1_PHASE_JOINING = 2,
  SAGR_CCL_LIVE_V1_PHASE_READY = 3,
  SAGR_CCL_LIVE_V1_PHASE_CLOSING = 4,
  SAGR_CCL_LIVE_V1_PHASE_ABORTED = 5,
  SAGR_CCL_LIVE_V1_PHASE_CLOSED = 6
} sagr_ccl_live_v1_phase_t;

typedef struct sagr_ccl_live_v1_process_identity {
  uint32_t struct_size;
  int32_t pid;
  uint32_t uid;
  uint32_t gid;
  uint64_t start_time_ticks;
  uint8_t reserved[16];
} sagr_ccl_live_v1_process_identity_t;

/*
 * A group abort is independent of carrier records. context_sequence is an
 * opaque caller-provided collective sequence (zero is valid before a
 * collective starts). The first complete record accepted by the broker is
 * immutable and is broadcast byte-for-byte to every authenticated rank.
 */
typedef struct sagr_ccl_live_v1_abort {
  uint32_t struct_size;
  uint32_t flags;
  sagr_ccl_v1_group_identity_t group;
  uint64_t context_sequence;
  uint32_t reporter_rank;
  uint32_t failed_rank;
  int32_t status;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_ccl_live_v1_abort_t;

typedef struct sagr_ccl_live_v1_rank_info {
  uint32_t struct_size;
  uint32_t phase;
  uint32_t self_rank;
  uint32_t world_size;
  int32_t control_socket;
  uint32_t reserved0;
  sagr_ccl_v1_group_identity_t group;
  /* Borrowed FDs. self_rank is always -1; every other rank is connected. */
  int32_t peer_sockets[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint8_t reserved[16];
} sagr_ccl_live_v1_rank_info_t;

typedef struct sagr_ccl_live_v1_broker_info {
  uint32_t struct_size;
  uint32_t phase;
  uint32_t world_size;
  uint32_t prepared_mask;
  uint32_t bound_mask;
  uint32_t joined_mask;
  uint32_t ready_mask;
  uint32_t departed_mask;
  uint32_t close_pending_mask;
  uint32_t abort_pending_mask;
  sagr_ccl_live_v1_process_identity_t owner;
  uint8_t reserved[16];
} sagr_ccl_live_v1_broker_info_t;

typedef struct sagr_ccl_live_v1_broker *sagr_ccl_live_v1_broker_t;
typedef struct sagr_ccl_live_v1_rank *sagr_ccl_live_v1_rank_t;

/* Broker and rank objects are caller-serialized. */

SAGR_API uint64_t sagr_ccl_live_v1_monotonic_time_ns(void);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_process_identity(
    int32_t pid, sagr_ccl_live_v1_process_identity_t *identity,
    uint32_t identity_size);

SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_create(
    const sagr_ccl_v1_group_identity_t *group,
    sagr_ccl_live_v1_broker_t *broker);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_info(
    sagr_ccl_live_v1_broker_t broker,
    sagr_ccl_live_v1_broker_info_t *info, uint32_t info_size);

/*
 * Call immediately before forking one rank. The returned capability FD is
 * caller-owned, initially CLOEXEC, and must be the only broker capability
 * explicitly passed across exec (for example with pass_fds, or by clearing
 * CLOEXEC in the child immediately before exec). The broker retains the other
 * endpoint. No pathname or bearer token exists.
 */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_prepare_rank(
    sagr_ccl_live_v1_broker_t broker, uint32_t rank,
    int *rank_capability_socket);
/* Call after fork with the PID/start-time identity observed by the supervisor. */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_bind_rank(
    sagr_ccl_live_v1_broker_t broker, uint32_t rank,
    const sagr_ccl_live_v1_process_identity_t *process);

/*
 * Authenticate all ranks, distribute pairwise FDs, collect table ACKs, and
 * release READY. The absolute CLOCK_MONOTONIC deadline is never extended.
 */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_rendezvous(
    sagr_ccl_live_v1_broker_t broker, uint64_t absolute_deadline_ns);

/* Nonblocking control-plane progress after READY. */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_progress(
    sagr_ccl_live_v1_broker_t broker,
    sagr_ccl_live_v1_abort_t *first_error, uint32_t first_error_size);
/* Supervisor-originated group failure; reporter_rank is NO_RANK. */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_abort(
    sagr_ccl_live_v1_broker_t broker, uint32_t failed_rank,
    sagr_ccl_v1_status_t reason, uint64_t context_sequence);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_first_error(
    sagr_ccl_live_v1_broker_t broker,
    sagr_ccl_live_v1_abort_t *first_error, uint32_t first_error_size);
SAGR_API void sagr_ccl_live_v1_broker_destroy(
    sagr_ccl_live_v1_broker_t *broker);

/*
 * Takes ownership of capability_socket on every return path. expected_broker
 * must be the exact PID/start-time identity that created the socketpair.
 */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_join(
    int capability_socket, const sagr_ccl_v1_group_identity_t *group,
    uint32_t self_rank,
    const sagr_ccl_live_v1_process_identity_t *expected_broker,
    uint64_t absolute_deadline_ns, sagr_ccl_live_v1_rank_t *rank);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_info(
    sagr_ccl_live_v1_rank_t rank, sagr_ccl_live_v1_rank_info_t *info,
    uint32_t info_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_report_abort(
    sagr_ccl_live_v1_rank_t rank, uint32_t failed_rank,
    sagr_ccl_v1_status_t reason, uint64_t context_sequence);
/* BUSY means no broker message is pending. */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_poll_abort(
    sagr_ccl_live_v1_rank_t rank, sagr_ccl_live_v1_abort_t *first_error,
    uint32_t first_error_size);
/*
 * Graceful group departure is collective. The call sends LEAVE and waits for
 * the broker to observe every rank before accepting CLOSED. A control HUP
 * before that point is a group PEER_LOST failure, never a successful close.
 */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_close(
    sagr_ccl_live_v1_rank_t rank, uint64_t absolute_deadline_ns);
SAGR_API void sagr_ccl_live_v1_rank_destroy(sagr_ccl_live_v1_rank_t *rank);

#ifdef __cplusplus
}
#endif

#endif
