/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_CCL_V1_H
#define SELF_AMDGPU_RUNTIME_CCL_V1_H

#include <stdint.h>

#include <self_amdgpu_runtime/export.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAGR_CCL_V1_PROTOCOL_MAJOR UINT16_C(1)
#define SAGR_CCL_V1_PROTOCOL_MINOR UINT16_C(0)
#define SAGR_CCL_V1_UUID_BYTES UINT32_C(16)
#define SAGR_CCL_V1_SHA256_BYTES UINT32_C(32)
#define SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES UINT32_C(160)
#define SAGR_CCL_V1_MIN_WORLD_SIZE UINT32_C(2)
#define SAGR_CCL_V1_MAX_WORLD_SIZE UINT32_C(16)
#define SAGR_CCL_V1_MAX_PLAN_STEPS \
  (UINT32_C(2) * (SAGR_CCL_V1_MAX_WORLD_SIZE - UINT32_C(1)))
#define SAGR_CCL_V1_NO_RANK UINT32_MAX
#define SAGR_CCL_V1_NO_CHUNK UINT32_MAX

typedef int32_t sagr_ccl_v1_status_t;

enum {
  SAGR_CCL_V1_STATUS_SUCCESS = 0,
  SAGR_CCL_V1_STATUS_INVALID_ARGUMENT = 1,
  SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL = 2,
  SAGR_CCL_V1_STATUS_VERSION_MISMATCH = 3,
  SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH = 4,
  SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH = 5,
  SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH = 6,
  SAGR_CCL_V1_STATUS_NOT_SUPPORTED = 7,
  SAGR_CCL_V1_STATUS_PROTOCOL_ERROR = 8,
  SAGR_CCL_V1_STATUS_CHECKSUM_ERROR = 9,
  SAGR_CCL_V1_STATUS_OUT_OF_ORDER = 10,
  SAGR_CCL_V1_STATUS_BUSY = 11,
  SAGR_CCL_V1_STATUS_ABORTED = 12,
  SAGR_CCL_V1_STATUS_TIMED_OUT = 13,
  SAGR_CCL_V1_STATUS_PEER_LOST = 14,
  SAGR_CCL_V1_STATUS_CANCELLED = 15,
  SAGR_CCL_V1_STATUS_CLOSED = 16,
  SAGR_CCL_V1_STATUS_OUT_OF_RESOURCES = 17
};

typedef enum sagr_ccl_v1_operation {
  SAGR_CCL_V1_OPERATION_INVALID = 0,
  SAGR_CCL_V1_OPERATION_ALL_REDUCE = 1,
  SAGR_CCL_V1_OPERATION_ALL_GATHER = 2,
  SAGR_CCL_V1_OPERATION_REDUCE_SCATTER = 3,
  SAGR_CCL_V1_OPERATION_BROADCAST = 4,
  SAGR_CCL_V1_OPERATION_BARRIER = 5
} sagr_ccl_v1_operation_t;

typedef enum sagr_ccl_v1_reduction {
  SAGR_CCL_V1_REDUCTION_NONE = 0,
  SAGR_CCL_V1_REDUCTION_SUM = 1
} sagr_ccl_v1_reduction_t;

typedef enum sagr_ccl_v1_dtype {
  SAGR_CCL_V1_DTYPE_NONE = 0,
  SAGR_CCL_V1_DTYPE_BF16 = 1,
  SAGR_CCL_V1_DTYPE_FP32 = 2,
  SAGR_CCL_V1_DTYPE_UINT8 = 3,
  SAGR_CCL_V1_DTYPE_INT32 = 4,
  SAGR_CCL_V1_DTYPE_UINT32 = 5
} sagr_ccl_v1_dtype_t;

typedef enum sagr_ccl_v1_plan_phase {
  SAGR_CCL_V1_PLAN_PHASE_INVALID = 0,
  SAGR_CCL_V1_PLAN_PHASE_REDUCE_SCATTER = 1,
  SAGR_CCL_V1_PLAN_PHASE_ALL_GATHER = 2,
  SAGR_CCL_V1_PLAN_PHASE_BROADCAST = 3,
  SAGR_CCL_V1_PLAN_PHASE_BARRIER = 4
} sagr_ccl_v1_plan_phase_t;

typedef enum sagr_ccl_v1_plan_action {
  SAGR_CCL_V1_PLAN_ACTION_IDLE = 0,
  SAGR_CCL_V1_PLAN_ACTION_SEND = 1,
  SAGR_CCL_V1_PLAN_ACTION_RECEIVE = 2,
  SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE = 3
} sagr_ccl_v1_plan_action_t;

typedef enum sagr_ccl_v1_group_phase {
  SAGR_CCL_V1_GROUP_PHASE_UNINITIALIZED = 0,
  SAGR_CCL_V1_GROUP_PHASE_JOINING = 1,
  SAGR_CCL_V1_GROUP_PHASE_READY = 2,
  SAGR_CCL_V1_GROUP_PHASE_COLLECTING = 3,
  SAGR_CCL_V1_GROUP_PHASE_ACTIVE = 4,
  SAGR_CCL_V1_GROUP_PHASE_ABORTED = 5,
  SAGR_CCL_V1_GROUP_PHASE_CLOSED = 6
} sagr_ccl_v1_group_phase_t;

/*
 * A group identity is exact.  A reconnect creates a new epoch or group
 * generation; a stale identity is never repaired or wildcard-matched.
 */
typedef struct sagr_ccl_v1_group_identity {
  uint32_t struct_size;
  uint32_t flags;
  uint16_t protocol_major;
  uint16_t protocol_minor;
  uint32_t world_size;
  uint64_t epoch;
  uint64_t group_generation;
  uint8_t job_uuid[SAGR_CCL_V1_UUID_BYTES];
  uint8_t group_uuid[SAGR_CCL_V1_UUID_BYTES];
  uint8_t model_identity_sha256[SAGR_CCL_V1_SHA256_BYTES];
  uint8_t reserved[16];
} sagr_ccl_v1_group_identity_t;

/*
 * Counts are per-rank logical element counts:
 *   all-reduce:     input_count == output_count
 *   all-gather:     output_count == input_count * world_size
 *   reduce-scatter: input_count == output_count * world_size
 *   broadcast:      input_count == output_count
 *   barrier:        both counts are zero
 * Integer dtypes are exact-copy test carriers only.  SUM is accepted only for
 * BF16 and FP32 and must be executed by a device kernel, never by this module.
 */
typedef struct sagr_ccl_v1_descriptor {
  uint32_t struct_size;
  uint32_t flags;
  sagr_ccl_v1_group_identity_t group;
  uint64_t sequence;
  uint64_t input_count;
  uint64_t output_count;
  uint32_t rank;
  uint32_t operation;
  uint32_t reduction;
  uint32_t dtype;
  uint32_t root_rank;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_ccl_v1_descriptor_t;

/*
 * Step offsets address a logical collective workspace, not the caller's raw
 * input pointer.  Before executing a plan, an implementation stages each
 * all-gather input in [rank * input_count, (rank + 1) * input_count); other
 * operations stage input from offset zero.  A nonzero REDUCE_SCATTER receive
 * maps to exactly one device SUM over the received chunk.  BF16 performs
 *   dst = bf16_rne(fp32(dst) + fp32(src))
 * at every planner hop; it does not retain an FP32 accumulator across hops.
 * FP32 performs one FP32 binary addition at every hop.  Source bytes remain
 * unchanged, zero elements are a successful no-op, and host arithmetic is
 * forbidden.  On standalone reduce-scatter completion, rank r's output is
 * workspace chunk r, starting at r * output_count.  This metadata layer never
 * reads or reduces payload elements itself.
 */
typedef struct sagr_ccl_v1_plan_step {
  uint32_t struct_size;
  uint32_t phase;
  uint32_t action;
  uint32_t step_index;
  uint32_t send_rank;
  uint32_t receive_rank;
  uint32_t send_chunk;
  uint32_t receive_chunk;
  uint64_t send_offset_elements;
  uint64_t send_count_elements;
  uint64_t receive_offset_elements;
  uint64_t receive_count_elements;
  uint8_t reserved[16];
} sagr_ccl_v1_plan_step_t;

/*
 * The standalone state machine records only metadata and rank progress.  It
 * contains no payload pointer and performs no copy or arithmetic.  The value
 * is caller-owned storage but its fields are runtime-owned: callers initialize
 * it once and then mutate it only through the group_state functions below.
 * Every mutating entry point validates the complete phase/mask/sequence/abort
 * invariant before changing state and poisons a structurally valid corrupted
 * value into the ABORTED phase.
 */
typedef struct sagr_ccl_v1_group_state {
  uint32_t struct_size;
  uint32_t phase;
  sagr_ccl_v1_group_identity_t identity;
  uint64_t next_sequence;
  uint64_t active_sequence;
  uint32_t joined_mask;
  uint32_t begun_mask;
  uint32_t completed_mask;
  uint32_t abort_rank;
  int32_t abort_status;
  uint32_t reserved0;
  sagr_ccl_v1_descriptor_t active_descriptor;
  uint8_t reserved[16];
} sagr_ccl_v1_group_state_t;

typedef struct sagr_ccl_v1_group_snapshot {
  uint32_t struct_size;
  uint32_t phase;
  uint64_t next_sequence;
  uint64_t active_sequence;
  uint32_t joined_mask;
  uint32_t begun_mask;
  uint32_t completed_mask;
  uint32_t abort_rank;
  int32_t abort_status;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_ccl_v1_group_snapshot_t;

SAGR_API const char *sagr_ccl_v1_status_string(sagr_ccl_v1_status_t status);

SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_group_identity_init(
    sagr_ccl_v1_group_identity_t *identity, uint32_t identity_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_group_identity_validate(
    const sagr_ccl_v1_group_identity_t *identity);
SAGR_API int sagr_ccl_v1_group_identity_equal(
    const sagr_ccl_v1_group_identity_t *left,
    const sagr_ccl_v1_group_identity_t *right);

SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_init(
    sagr_ccl_v1_descriptor_t *descriptor, uint32_t descriptor_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_validate(
    const sagr_ccl_v1_descriptor_t *descriptor);
/* Canonical wire encoding is big-endian and ends with a CRC32C plus zero
 * reserved bytes.  Decode requires exactly SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES. */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_encode(
    const sagr_ccl_v1_descriptor_t *descriptor, uint8_t *wire,
    uint32_t wire_size);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_decode(
    const uint8_t *wire, uint32_t wire_size,
    sagr_ccl_v1_descriptor_t *descriptor, uint32_t descriptor_size);
/* Rank-independent canonical hash of one collective descriptor. */
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_sha256(
    const sagr_ccl_v1_descriptor_t *descriptor,
    uint8_t digest[SAGR_CCL_V1_SHA256_BYTES]);

SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_plan_required_steps(
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t *required_steps);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_plan_rank(
    const sagr_ccl_v1_descriptor_t *descriptor,
    sagr_ccl_v1_plan_step_t *steps, uint32_t step_capacity,
    uint32_t *step_count);

SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_group_state_init(
    sagr_ccl_v1_group_state_t *state, uint32_t state_size,
    const sagr_ccl_v1_group_identity_t *identity);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_group_state_join(
    sagr_ccl_v1_group_state_t *state,
    const sagr_ccl_v1_group_identity_t *identity, uint32_t rank);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_group_state_begin(
    sagr_ccl_v1_group_state_t *state,
    const sagr_ccl_v1_descriptor_t *descriptor);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_group_state_complete(
    sagr_ccl_v1_group_state_t *state, uint32_t rank, uint64_t sequence);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_group_state_abort(
    sagr_ccl_v1_group_state_t *state, uint32_t rank, uint64_t sequence,
    sagr_ccl_v1_status_t reason);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_group_state_close(
    sagr_ccl_v1_group_state_t *state);
SAGR_API sagr_ccl_v1_status_t sagr_ccl_v1_group_state_snapshot(
    const sagr_ccl_v1_group_state_t *state,
    sagr_ccl_v1_group_snapshot_t *snapshot, uint32_t snapshot_size);

#ifdef __cplusplus
}
#endif

#endif
