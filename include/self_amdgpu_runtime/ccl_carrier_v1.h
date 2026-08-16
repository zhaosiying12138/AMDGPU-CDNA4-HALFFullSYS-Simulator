/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_CCL_CARRIER_V1_H
#define SELF_AMDGPU_RUNTIME_CCL_CARRIER_V1_H

#include <stdint.h>

#include <self_amdgpu_runtime/ccl_v1.h>
#include <self_amdgpu_runtime/export.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAGR_CCL_V1_CARRIER_WIRE_BYTES UINT32_C(240)
#define SAGR_CCL_V1_CARRIER_MAX_PAYLOAD_BYTES UINT64_C(16777216)
#define SAGR_CCL_V1_MAX_CREDITS_PER_PEER UINT32_C(16)

typedef enum sagr_ccl_v1_carrier_slot_phase {
  SAGR_CCL_V1_CARRIER_SLOT_UNINITIALIZED = 0,
  SAGR_CCL_V1_CARRIER_SLOT_EMPTY = 1,
  SAGR_CCL_V1_CARRIER_SLOT_READY = 2,
  SAGR_CCL_V1_CARRIER_SLOT_CONSUMED = 3,
  SAGR_CCL_V1_CARRIER_SLOT_ABORTED = 4,
  SAGR_CCL_V1_CARRIER_SLOT_CLOSED = 5
} sagr_ccl_v1_carrier_slot_phase_t;

typedef enum sagr_ccl_v1_carrier_message_kind {
  SAGR_CCL_V1_CARRIER_MESSAGE_INVALID = 0,
  SAGR_CCL_V1_CARRIER_MESSAGE_DATA = 1,
  SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED = 2,
  SAGR_CCL_V1_CARRIER_MESSAGE_ABORT = 3
} sagr_ccl_v1_carrier_message_kind_t;

typedef enum sagr_ccl_v1_carrier_session_phase {
  SAGR_CCL_V1_CARRIER_SESSION_UNINITIALIZED = 0,
  SAGR_CCL_V1_CARRIER_SESSION_RUNNING = 1,
  SAGR_CCL_V1_CARRIER_SESSION_ABORTED = 2,
  SAGR_CCL_V1_CARRIER_SESSION_CLOSED = 3
} sagr_ccl_v1_carrier_session_phase_t;

/*
 * One plan transfer, bound to an exact collective and sealed payload.  This
 * layer copies and verifies bytes only; it never performs reduction arithmetic.
 */
typedef struct sagr_ccl_v1_carrier_record {
  uint32_t struct_size;
  uint32_t flags;
  sagr_ccl_v1_group_identity_t group;
  uint8_t descriptor_sha256[SAGR_CCL_V1_SHA256_BYTES];
  uint64_t sequence;
  uint64_t slot_generation;
  uint64_t payload_bytes;
  uint32_t kind;
  uint32_t phase;
  uint32_t step_index;
  uint32_t chunk_index;
  uint32_t source_rank;
  uint32_t destination_rank;
  uint32_t slot_index;
  uint32_t payload_crc32c;
  int32_t status;
  uint32_t failed_rank;
  uint32_t reserved0;
  uint8_t reserved[12];
} sagr_ccl_v1_carrier_record_t;

typedef struct sagr_ccl_v1_credit_state {
  uint32_t struct_size;
  uint32_t flags;
  sagr_ccl_v1_group_identity_t group;
  uint32_t self_rank;
  uint32_t credits_per_peer;
  uint32_t occupied_mask[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t reserved0;
  uint64_t next_generation[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint64_t slot_generation[SAGR_CCL_V1_MAX_WORLD_SIZE]
                          [SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
  uint8_t reserved[32];
} sagr_ccl_v1_credit_state_t;

typedef struct sagr_ccl_v1_carrier_slot_state {
  uint32_t struct_size;
  uint32_t phase;
  sagr_ccl_v1_carrier_record_t record;
  uint32_t abort_status;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_ccl_v1_carrier_slot_state_t;

typedef struct sagr_ccl_v1_carrier_session
    *sagr_ccl_v1_carrier_session_t;

typedef struct sagr_ccl_v1_carrier_session_info {
  uint32_t struct_size;
  uint32_t phase;
  uint32_t self_rank;
  uint32_t world_size;
  uint32_t credits_per_peer;
  uint32_t sender_inflight;
  uint32_t receiver_ready;
  uint32_t receiver_consumed;
  int32_t first_error;
  uint32_t failed_rank;
  uint8_t reserved[16];
} sagr_ccl_v1_carrier_session_info_t;

SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_init(
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_validate(
    const sagr_ccl_v1_carrier_record_t *record);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_encode(
    const sagr_ccl_v1_carrier_record_t *record, uint8_t *wire,
    uint32_t wire_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_decode(
    const uint8_t *wire, uint32_t wire_size,
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_validate_descriptor(
    const sagr_ccl_v1_carrier_record_t *record,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t expected_step_index,
    uint32_t authenticated_peer_rank);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_from_plan(
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t step_index,
    uint32_t source_rank, uint32_t kind, uint32_t slot_index,
    uint64_t slot_generation, uint32_t payload_crc32c,
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_abort_record(
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t reporter_rank,
    uint32_t failed_rank, sagr_ccl_v1_status_t reason,
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size);

SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_credit_state_init(
    sagr_ccl_v1_credit_state_t *state, uint32_t state_size,
    const sagr_ccl_v1_group_identity_t *group, uint32_t self_rank,
    uint32_t credits_per_peer);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_credit_acquire(
    sagr_ccl_v1_credit_state_t *state, uint32_t destination_rank,
    uint32_t *slot_index, uint64_t *slot_generation);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_credit_release(
    sagr_ccl_v1_credit_state_t *state, uint32_t destination_rank,
    uint32_t slot_index, uint64_t slot_generation);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_credit_state_close(
    sagr_ccl_v1_credit_state_t *state);

SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_init(
    sagr_ccl_v1_carrier_slot_state_t *slot, uint32_t slot_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_publish(
    sagr_ccl_v1_carrier_slot_state_t *slot,
    const sagr_ccl_v1_carrier_record_t *record);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_consume(
    sagr_ccl_v1_carrier_slot_state_t *slot,
    const sagr_ccl_v1_carrier_record_t *record);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_release(
    sagr_ccl_v1_carrier_slot_state_t *slot,
    const sagr_ccl_v1_carrier_record_t *record);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_abort(
    sagr_ccl_v1_carrier_slot_state_t *slot,
    sagr_ccl_v1_status_t reason);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_close(
    sagr_ccl_v1_carrier_slot_state_t *slot);

SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_payload_create(
    const void *bytes, uint64_t byte_count, int *descriptor,
    uint32_t *payload_crc32c);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_payload_validate(
    int descriptor, uint64_t byte_count, uint32_t payload_crc32c);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_payload_copy(
    int descriptor, uint64_t byte_count, uint32_t payload_crc32c,
    void *destination, uint64_t destination_capacity);

/* One-record nonblocking operations: EAGAIN maps to BUSY. */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_send(
    int socket_descriptor, const sagr_ccl_v1_carrier_record_t *record,
    int payload_descriptor);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_receive(
    int socket_descriptor, sagr_ccl_v1_carrier_record_t *record,
    uint32_t record_size, int *payload_descriptor);

/*
 * The session is the production ownership boundary.  It owns every payload FD
 * and credit from prepare through CONSUMED, is caller-serialized, and poisons
 * the complete group epoch on an ambiguous send or peer/protocol failure.
 */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_create(
    const sagr_ccl_v1_group_identity_t *group, uint32_t self_rank,
    uint32_t credits_per_peer, sagr_ccl_v1_carrier_session_t *session);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_prepare_data(
    sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t step_index,
    const void *payload, uint64_t payload_bytes,
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_send_data(
    sagr_ccl_v1_carrier_session_t session, int socket_descriptor,
    const sagr_ccl_v1_carrier_record_t *record);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_receive(
    sagr_ccl_v1_carrier_session_t session, int socket_descriptor,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t expected_step_index,
    uint32_t authenticated_peer_rank, sagr_ccl_v1_carrier_record_t *record,
    uint32_t record_size);
/*
 * Consume copies DATA into caller-owned staging and returns a CONSUMED record;
 * it does not transmit the acknowledgement.  For REDUCE_SCATTER, staging must
 * be immutable and disjoint from the workspace destination.  The caller must
 * complete the device SUM synchronously before send_consumed.  A validation,
 * copy, or device failure aborts the group and must never acknowledge DATA.
 * A zero-byte step is consumed and acknowledged without a device dispatch.
 */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_consume(
    sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t expected_step_index,
    const sagr_ccl_v1_carrier_record_t *data_record, void *destination,
    uint64_t destination_capacity,
    sagr_ccl_v1_carrier_record_t *consumed_record,
    uint32_t consumed_record_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_send_consumed(
    sagr_ccl_v1_carrier_session_t session, int socket_descriptor,
    const sagr_ccl_v1_carrier_record_t *record);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_abort(
    sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t failed_rank,
    sagr_ccl_v1_status_t reason, sagr_ccl_v1_carrier_record_t *abort_record,
    uint32_t abort_record_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_send_abort(
    sagr_ccl_v1_carrier_session_t session, int socket_descriptor,
    const sagr_ccl_v1_carrier_record_t *abort_record);
/*
 * Returns the immutable first-error record.  A relaying rank forwards this
 * exact record: source_rank continues to identify the original reporter.
 */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_get_abort(
    sagr_ccl_v1_carrier_session_t session,
    sagr_ccl_v1_carrier_record_t *abort_record,
    uint32_t abort_record_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_info(
    sagr_ccl_v1_carrier_session_t session,
    sagr_ccl_v1_carrier_session_info_t *info, uint32_t info_size);
SAGR_API void sagr_ccl_v1_carrier_session_destroy(
    sagr_ccl_v1_carrier_session_t *session);

#ifdef __cplusplus
}
#endif

#endif
