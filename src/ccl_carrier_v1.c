/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <self_amdgpu_runtime/ccl_carrier_v1.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define CARRIER_MAGIC UINT32_C(0x53434343)
#define CARRIER_CRC_OFFSET UINT32_C(196)
#define CARRIER_RESERVED_OFFSET UINT32_C(200)
#define CREDIT_CLOSED_FLAG UINT32_C(1)
#define CARRIER_SESSION_MAGIC UINT64_C(0x5341475243434c31)

typedef enum carrier_owned_phase {
  CARRIER_OWNED_EMPTY = 0,
  CARRIER_OWNED_PREPARED = 1,
  CARRIER_OWNED_SENT = 2,
  CARRIER_OWNED_READY = 3,
  CARRIER_OWNED_CONSUMED = 4
} carrier_owned_phase_t;

typedef struct carrier_owned_transfer {
  uint32_t phase;
  int descriptor;
  sagr_ccl_v1_carrier_record_t record;
} carrier_owned_transfer_t;

struct sagr_ccl_v1_carrier_session {
  uint64_t magic;
  uint32_t phase;
  uint32_t self_rank;
  uint32_t credits_per_peer;
  int32_t first_error;
  uint32_t failed_rank;
  sagr_ccl_v1_group_identity_t group;
  sagr_ccl_v1_credit_state_t credits;
  sagr_ccl_v1_carrier_record_t abort_record;
  carrier_owned_transfer_t
      sender[SAGR_CCL_V1_MAX_WORLD_SIZE]
            [SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
  carrier_owned_transfer_t
      receiver[SAGR_CCL_V1_MAX_WORLD_SIZE]
              [SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
  uint64_t receiver_last_generation[SAGR_CCL_V1_MAX_WORLD_SIZE]
                                   [SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
};

static int bytes_zero(const uint8_t *bytes, size_t count) {
  size_t index;
  for (index = 0U; index < count; ++index) {
    if (bytes[index] != 0U) {
      return 0;
    }
  }
  return 1;
}

static int bytes_nonzero(const uint8_t *bytes, size_t count) {
  return !bytes_zero(bytes, count);
}

static void put_u16(uint8_t *destination, uint16_t value) {
  destination[0] = (uint8_t)(value >> 8U);
  destination[1] = (uint8_t)value;
}

static void put_u32(uint8_t *destination, uint32_t value) {
  destination[0] = (uint8_t)(value >> 24U);
  destination[1] = (uint8_t)(value >> 16U);
  destination[2] = (uint8_t)(value >> 8U);
  destination[3] = (uint8_t)value;
}

static void put_u64(uint8_t *destination, uint64_t value) {
  put_u32(destination, (uint32_t)(value >> 32U));
  put_u32(destination + 4, (uint32_t)value);
}

static uint16_t get_u16(const uint8_t *source) {
  return (uint16_t)(((uint16_t)source[0] << 8U) | source[1]);
}

static uint32_t get_u32(const uint8_t *source) {
  return ((uint32_t)source[0] << 24U) | ((uint32_t)source[1] << 16U) |
         ((uint32_t)source[2] << 8U) | source[3];
}

static uint64_t get_u64(const uint8_t *source) {
  return ((uint64_t)get_u32(source) << 32U) | get_u32(source + 4);
}

static uint32_t crc32c(const uint8_t *bytes, size_t count) {
  uint32_t crc = UINT32_MAX;
  size_t index;
  for (index = 0U; index < count; ++index) {
    uint32_t value = crc ^ bytes[index];
    unsigned bit;
    for (bit = 0U; bit < 8U; ++bit) {
      const uint32_t mask = (uint32_t)-(int32_t)(value & UINT32_C(1));
      value = (value >> 1U) ^ (UINT32_C(0x82f63b78) & mask);
    }
    crc = value;
  }
  return ~crc;
}

static uint32_t crc32c_extend(uint32_t crc, const uint8_t *bytes,
                              size_t count) {
  size_t index;
  for (index = 0U; index < count; ++index) {
    uint32_t value = crc ^ bytes[index];
    unsigned bit;
    for (bit = 0U; bit < 8U; ++bit) {
      const uint32_t mask = (uint32_t)-(int32_t)(value & UINT32_C(1));
      value = (value >> 1U) ^ (UINT32_C(0x82f63b78) & mask);
    }
    crc = value;
  }
  return crc;
}

static uint32_t credit_mask(uint32_t credits) {
  return credits == 32U ? UINT32_MAX
                        : (UINT32_C(1) << credits) - UINT32_C(1);
}

static int valid_plan_phase(uint32_t phase) {
  return phase == SAGR_CCL_V1_PLAN_PHASE_REDUCE_SCATTER ||
         phase == SAGR_CCL_V1_PLAN_PHASE_ALL_GATHER ||
         phase == SAGR_CCL_V1_PLAN_PHASE_BROADCAST ||
         phase == SAGR_CCL_V1_PLAN_PHASE_BARRIER;
}

static int valid_abort_status(sagr_ccl_v1_status_t status) {
  return status == SAGR_CCL_V1_STATUS_VERSION_MISMATCH ||
         status == SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH ||
         status == SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH ||
         status == SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH ||
         status == SAGR_CCL_V1_STATUS_NOT_SUPPORTED ||
         status == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR ||
         status == SAGR_CCL_V1_STATUS_CHECKSUM_ERROR ||
         status == SAGR_CCL_V1_STATUS_OUT_OF_ORDER ||
         status == SAGR_CCL_V1_STATUS_TIMED_OUT ||
         status == SAGR_CCL_V1_STATUS_PEER_LOST ||
         status == SAGR_CCL_V1_STATUS_CANCELLED ||
         status == SAGR_CCL_V1_STATUS_OUT_OF_RESOURCES;
}

static sagr_ccl_v1_status_t status_from_resource_errno(int error) {
  switch (error) {
    case EAGAIN:
      return SAGR_CCL_V1_STATUS_BUSY;
    case ENOSYS:
      return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
    case EMFILE:
    case ENFILE:
    case ENOMEM:
    case ENOSPC:
      return SAGR_CCL_V1_STATUS_OUT_OF_RESOURCES;
    default:
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
}

static uint32_t dtype_bytes(uint32_t dtype) {
  switch (dtype) {
    case SAGR_CCL_V1_DTYPE_BF16:
      return 2U;
    case SAGR_CCL_V1_DTYPE_FP32:
    case SAGR_CCL_V1_DTYPE_INT32:
    case SAGR_CCL_V1_DTYPE_UINT32:
      return 4U;
    case SAGR_CCL_V1_DTYPE_UINT8:
      return 1U;
    default:
      return 0U;
  }
}

static int multiply_u64(uint64_t left, uint64_t right, uint64_t *result) {
  if (result == NULL || (left != 0U && right > UINT64_MAX / left)) {
    return 0;
  }
  *result = left * right;
  return 1;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_init(
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size) {
  if (record == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (record_size < sizeof(*record)) {
    if (record_size >= sizeof(record->struct_size)) {
      record->struct_size = (uint32_t)sizeof(*record);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  memset(record, 0, sizeof(*record));
  record->struct_size = (uint32_t)sizeof(*record);
  record->failed_rank = SAGR_CCL_V1_NO_RANK;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_validate(
    const sagr_ccl_v1_carrier_record_t *record) {
  if (record == NULL || record->struct_size != sizeof(*record)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (record->flags != 0U || record->reserved0 != 0U ||
      !bytes_zero(record->reserved, sizeof(record->reserved))) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (sagr_ccl_v1_group_identity_validate(&record->group) !=
      SAGR_CCL_V1_STATUS_SUCCESS) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  if (!bytes_nonzero(record->descriptor_sha256,
                     sizeof(record->descriptor_sha256)) ||
      record->sequence == 0U || record->sequence == UINT64_MAX ||
      record->payload_bytes > SAGR_CCL_V1_CARRIER_MAX_PAYLOAD_BYTES ||
      record->source_rank >= record->group.world_size) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  switch (record->kind) {
    case SAGR_CCL_V1_CARRIER_MESSAGE_DATA:
      if (record->slot_generation == 0U ||
          record->slot_generation == UINT64_MAX ||
          !valid_plan_phase(record->phase) ||
          record->step_index >= SAGR_CCL_V1_MAX_PLAN_STEPS ||
          record->destination_rank >= record->group.world_size ||
          record->source_rank == record->destination_rank ||
          record->slot_index >= SAGR_CCL_V1_MAX_CREDITS_PER_PEER ||
          record->status != SAGR_CCL_V1_STATUS_SUCCESS ||
          record->failed_rank != SAGR_CCL_V1_NO_RANK ||
          (record->payload_bytes == 0U && record->payload_crc32c != 0U)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED:
      if (record->slot_generation == 0U ||
          record->slot_generation == UINT64_MAX ||
          !valid_plan_phase(record->phase) ||
          record->step_index >= SAGR_CCL_V1_MAX_PLAN_STEPS ||
          record->destination_rank >= record->group.world_size ||
          record->source_rank == record->destination_rank ||
          record->slot_index >= SAGR_CCL_V1_MAX_CREDITS_PER_PEER ||
          record->payload_bytes != 0U || record->payload_crc32c != 0U ||
          record->status != SAGR_CCL_V1_STATUS_SUCCESS ||
          record->failed_rank != SAGR_CCL_V1_NO_RANK) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_CARRIER_MESSAGE_ABORT:
      if (record->slot_generation != 0U || record->payload_bytes != 0U ||
          record->payload_crc32c != 0U ||
          record->phase != SAGR_CCL_V1_PLAN_PHASE_INVALID ||
          record->step_index != SAGR_CCL_V1_NO_CHUNK ||
          record->chunk_index != SAGR_CCL_V1_NO_CHUNK ||
          record->destination_rank != SAGR_CCL_V1_NO_RANK ||
          record->slot_index != SAGR_CCL_V1_NO_CHUNK ||
          !valid_abort_status(record->status) ||
          (record->failed_rank != SAGR_CCL_V1_NO_RANK &&
           record->failed_rank >= record->group.world_size)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    default:
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_encode(
    const sagr_ccl_v1_carrier_record_t *record, uint8_t *wire,
    uint32_t wire_size) {
  sagr_ccl_v1_status_t status = sagr_ccl_v1_carrier_record_validate(record);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (wire == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (wire_size < SAGR_CCL_V1_CARRIER_WIRE_BYTES) {
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  memset(wire, 0, SAGR_CCL_V1_CARRIER_WIRE_BYTES);
  put_u32(wire, CARRIER_MAGIC);
  put_u16(wire + 4, record->group.protocol_major);
  put_u16(wire + 6, record->group.protocol_minor);
  put_u32(wire + 8, SAGR_CCL_V1_CARRIER_WIRE_BYTES);
  put_u32(wire + 12, record->flags);
  put_u32(wire + 16, record->group.world_size);
  put_u32(wire + 20, record->kind);
  put_u32(wire + 24, record->phase);
  put_u32(wire + 28, record->step_index);
  put_u32(wire + 32, record->chunk_index);
  put_u32(wire + 36, record->source_rank);
  put_u32(wire + 40, record->destination_rank);
  put_u32(wire + 44, record->slot_index);
  put_u32(wire + 48, (uint32_t)record->status);
  put_u32(wire + 52, record->failed_rank);
  put_u64(wire + 56, record->sequence);
  put_u64(wire + 64, record->slot_generation);
  put_u64(wire + 72, record->payload_bytes);
  put_u64(wire + 80, record->group.epoch);
  put_u64(wire + 88, record->group.group_generation);
  memcpy(wire + 96, record->group.job_uuid, sizeof(record->group.job_uuid));
  memcpy(wire + 112, record->group.group_uuid,
         sizeof(record->group.group_uuid));
  memcpy(wire + 128, record->group.model_identity_sha256,
         sizeof(record->group.model_identity_sha256));
  memcpy(wire + 160, record->descriptor_sha256,
         sizeof(record->descriptor_sha256));
  put_u32(wire + 192, record->payload_crc32c);
  put_u32(wire + CARRIER_CRC_OFFSET, crc32c(wire, CARRIER_CRC_OFFSET));
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_decode(
    const uint8_t *wire, uint32_t wire_size,
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size) {
  sagr_ccl_v1_status_t status;
  if (wire == NULL || record == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (wire_size != SAGR_CCL_V1_CARRIER_WIRE_BYTES) {
    return wire_size < SAGR_CCL_V1_CARRIER_WIRE_BYTES
               ? SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL
               : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (record_size < sizeof(*record)) {
    if (record_size >= sizeof(record->struct_size)) {
      record->struct_size = (uint32_t)sizeof(*record);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  if (get_u32(wire) != CARRIER_MAGIC ||
      get_u32(wire + 8) != SAGR_CCL_V1_CARRIER_WIRE_BYTES ||
      !bytes_zero(wire + CARRIER_RESERVED_OFFSET,
                  SAGR_CCL_V1_CARRIER_WIRE_BYTES - CARRIER_RESERVED_OFFSET)) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (get_u16(wire + 4) != SAGR_CCL_V1_PROTOCOL_MAJOR ||
      get_u16(wire + 6) != SAGR_CCL_V1_PROTOCOL_MINOR) {
    return SAGR_CCL_V1_STATUS_VERSION_MISMATCH;
  }
  if (get_u32(wire + CARRIER_CRC_OFFSET) !=
      crc32c(wire, CARRIER_CRC_OFFSET)) {
    return SAGR_CCL_V1_STATUS_CHECKSUM_ERROR;
  }
  status = sagr_ccl_v1_carrier_record_init(record, record_size);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  status = sagr_ccl_v1_group_identity_init(
      &record->group, (uint32_t)sizeof(record->group));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  record->flags = get_u32(wire + 12);
  record->group.world_size = get_u32(wire + 16);
  record->kind = get_u32(wire + 20);
  record->phase = get_u32(wire + 24);
  record->step_index = get_u32(wire + 28);
  record->chunk_index = get_u32(wire + 32);
  record->source_rank = get_u32(wire + 36);
  record->destination_rank = get_u32(wire + 40);
  record->slot_index = get_u32(wire + 44);
  record->status = (int32_t)get_u32(wire + 48);
  record->failed_rank = get_u32(wire + 52);
  record->sequence = get_u64(wire + 56);
  record->slot_generation = get_u64(wire + 64);
  record->payload_bytes = get_u64(wire + 72);
  record->group.epoch = get_u64(wire + 80);
  record->group.group_generation = get_u64(wire + 88);
  memcpy(record->group.job_uuid, wire + 96,
         sizeof(record->group.job_uuid));
  memcpy(record->group.group_uuid, wire + 112,
         sizeof(record->group.group_uuid));
  memcpy(record->group.model_identity_sha256, wire + 128,
         sizeof(record->group.model_identity_sha256));
  memcpy(record->descriptor_sha256, wire + 160,
         sizeof(record->descriptor_sha256));
  record->payload_crc32c = get_u32(wire + 192);
  return sagr_ccl_v1_carrier_record_validate(record);
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_validate_descriptor(
    const sagr_ccl_v1_carrier_record_t *record,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t expected_step_index,
    uint32_t authenticated_peer_rank) {
  sagr_ccl_v1_descriptor_t source_descriptor;
  sagr_ccl_v1_descriptor_t destination_descriptor;
  sagr_ccl_v1_plan_step_t steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
  sagr_ccl_v1_plan_step_t destination_steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
  uint8_t descriptor_sha256[SAGR_CCL_V1_SHA256_BYTES];
  uint64_t expected_payload_bytes = 0U;
  uint32_t step_count = 0U;
  uint32_t destination_step_count = 0U;
  sagr_ccl_v1_status_t status =
      sagr_ccl_v1_carrier_record_validate(record);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  status = sagr_ccl_v1_descriptor_sha256(descriptor, descriptor_sha256);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (!sagr_ccl_v1_group_identity_equal(&record->group,
                                         &descriptor->group)) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  if (record->sequence != descriptor->sequence ||
      memcmp(record->descriptor_sha256, descriptor_sha256,
             sizeof(descriptor_sha256)) != 0) {
    return SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH;
  }
  if (record->kind == SAGR_CCL_V1_CARRIER_MESSAGE_ABORT) {
    return expected_step_index == SAGR_CCL_V1_NO_CHUNK &&
                   authenticated_peer_rank < descriptor->group.world_size &&
                   authenticated_peer_rank != descriptor->rank
               ? SAGR_CCL_V1_STATUS_SUCCESS
               : SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH;
  }
  if (expected_step_index != record->step_index ||
      authenticated_peer_rank >= descriptor->group.world_size) {
    return SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH;
  }
  if ((record->kind == SAGR_CCL_V1_CARRIER_MESSAGE_DATA &&
       (descriptor->rank != record->destination_rank ||
        authenticated_peer_rank != record->source_rank)) ||
      (record->kind == SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED &&
       (descriptor->rank != record->source_rank ||
        authenticated_peer_rank != record->destination_rank))) {
    return SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH;
  }
  source_descriptor = *descriptor;
  source_descriptor.rank = record->source_rank;
  status = sagr_ccl_v1_plan_rank(&source_descriptor, steps,
                                 SAGR_CCL_V1_MAX_PLAN_STEPS, &step_count);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (record->step_index >= step_count ||
      (steps[record->step_index].action != SAGR_CCL_V1_PLAN_ACTION_SEND &&
       steps[record->step_index].action !=
           SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE) ||
      steps[record->step_index].send_rank != record->destination_rank ||
      steps[record->step_index].phase != record->phase ||
      steps[record->step_index].send_chunk != record->chunk_index) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  destination_descriptor = *descriptor;
  destination_descriptor.rank = record->destination_rank;
  status = sagr_ccl_v1_plan_rank(
      &destination_descriptor, destination_steps, SAGR_CCL_V1_MAX_PLAN_STEPS,
      &destination_step_count);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (record->step_index >= destination_step_count ||
      (destination_steps[record->step_index].action !=
           SAGR_CCL_V1_PLAN_ACTION_RECEIVE &&
       destination_steps[record->step_index].action !=
           SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE) ||
      destination_steps[record->step_index].receive_rank !=
          record->source_rank ||
      destination_steps[record->step_index].phase != record->phase ||
      destination_steps[record->step_index].receive_chunk !=
          record->chunk_index ||
      destination_steps[record->step_index].receive_offset_elements !=
          steps[record->step_index].send_offset_elements ||
      destination_steps[record->step_index].receive_count_elements !=
          steps[record->step_index].send_count_elements) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (!multiply_u64(steps[record->step_index].send_count_elements,
                    dtype_bytes(descriptor->dtype),
                    &expected_payload_bytes) ||
      expected_payload_bytes > SAGR_CCL_V1_CARRIER_MAX_PAYLOAD_BYTES) {
    return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
  }
  if (record->kind == SAGR_CCL_V1_CARRIER_MESSAGE_DATA &&
      record->payload_bytes != expected_payload_bytes) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_record_from_plan(
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t step_index,
    uint32_t source_rank, uint32_t kind, uint32_t slot_index,
    uint64_t slot_generation, uint32_t payload_crc32c,
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size) {
  sagr_ccl_v1_descriptor_t source_descriptor;
  sagr_ccl_v1_descriptor_t destination_descriptor;
  sagr_ccl_v1_plan_step_t steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
  sagr_ccl_v1_plan_step_t destination_steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
  uint64_t payload_bytes = 0U;
  uint32_t step_count = 0U;
  uint32_t destination_step_count = 0U;
  sagr_ccl_v1_status_t status;
  if (descriptor == NULL || record == NULL ||
      (kind != SAGR_CCL_V1_CARRIER_MESSAGE_DATA &&
       kind != SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  status = sagr_ccl_v1_descriptor_validate(descriptor);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (source_rank >= descriptor->group.world_size ||
      (kind == SAGR_CCL_V1_CARRIER_MESSAGE_DATA &&
       descriptor->rank != source_rank)) {
    return SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH;
  }
  source_descriptor = *descriptor;
  source_descriptor.rank = source_rank;
  status = sagr_ccl_v1_plan_rank(&source_descriptor, steps,
                                 SAGR_CCL_V1_MAX_PLAN_STEPS, &step_count);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (step_index >= step_count ||
      (steps[step_index].action != SAGR_CCL_V1_PLAN_ACTION_SEND &&
       steps[step_index].action != SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE)) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  if (kind == SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED &&
      descriptor->rank != steps[step_index].send_rank) {
    return SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH;
  }
  destination_descriptor = *descriptor;
  destination_descriptor.rank = steps[step_index].send_rank;
  status = sagr_ccl_v1_plan_rank(
      &destination_descriptor, destination_steps, SAGR_CCL_V1_MAX_PLAN_STEPS,
      &destination_step_count);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (step_index >= destination_step_count ||
      (destination_steps[step_index].action !=
           SAGR_CCL_V1_PLAN_ACTION_RECEIVE &&
       destination_steps[step_index].action !=
           SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE) ||
      destination_steps[step_index].receive_rank != source_rank ||
      destination_steps[step_index].phase != steps[step_index].phase ||
      destination_steps[step_index].receive_chunk !=
          steps[step_index].send_chunk ||
      destination_steps[step_index].receive_offset_elements !=
          steps[step_index].send_offset_elements ||
      destination_steps[step_index].receive_count_elements !=
          steps[step_index].send_count_elements) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (!multiply_u64(steps[step_index].send_count_elements,
                    dtype_bytes(descriptor->dtype), &payload_bytes) ||
      payload_bytes > SAGR_CCL_V1_CARRIER_MAX_PAYLOAD_BYTES) {
    return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
  }
  status = sagr_ccl_v1_carrier_record_init(record, record_size);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  record->group = descriptor->group;
  status = sagr_ccl_v1_descriptor_sha256(descriptor,
                                         record->descriptor_sha256);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  record->sequence = descriptor->sequence;
  record->slot_generation = slot_generation;
  record->payload_bytes =
      kind == SAGR_CCL_V1_CARRIER_MESSAGE_DATA ? payload_bytes : 0U;
  record->kind = kind;
  record->phase = steps[step_index].phase;
  record->step_index = step_index;
  record->chunk_index = steps[step_index].send_chunk;
  record->source_rank = source_rank;
  record->destination_rank = steps[step_index].send_rank;
  record->slot_index = slot_index;
  record->payload_crc32c =
      kind == SAGR_CCL_V1_CARRIER_MESSAGE_DATA ? payload_crc32c : 0U;
  return sagr_ccl_v1_carrier_record_validate(record);
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_abort_record(
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t reporter_rank,
    uint32_t failed_rank, sagr_ccl_v1_status_t reason,
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size) {
  sagr_ccl_v1_status_t status;
  if (descriptor == NULL || record == NULL ||
      reporter_rank >= descriptor->group.world_size ||
      (failed_rank != SAGR_CCL_V1_NO_RANK &&
       failed_rank >= descriptor->group.world_size) ||
      !valid_abort_status(reason)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  status = sagr_ccl_v1_descriptor_validate(descriptor);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  status = sagr_ccl_v1_carrier_record_init(record, record_size);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  record->group = descriptor->group;
  status = sagr_ccl_v1_descriptor_sha256(descriptor,
                                         record->descriptor_sha256);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  record->sequence = descriptor->sequence;
  record->kind = SAGR_CCL_V1_CARRIER_MESSAGE_ABORT;
  record->phase = SAGR_CCL_V1_PLAN_PHASE_INVALID;
  record->step_index = SAGR_CCL_V1_NO_CHUNK;
  record->chunk_index = SAGR_CCL_V1_NO_CHUNK;
  record->source_rank = reporter_rank;
  record->destination_rank = SAGR_CCL_V1_NO_RANK;
  record->slot_index = SAGR_CCL_V1_NO_CHUNK;
  record->status = reason;
  record->failed_rank = failed_rank;
  return sagr_ccl_v1_carrier_record_validate(record);
}

static sagr_ccl_v1_status_t validate_credit_state(
    const sagr_ccl_v1_credit_state_t *state) {
  uint32_t rank;
  uint32_t mask;
  if (state == NULL || state->struct_size != sizeof(*state)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (state->flags == CREDIT_CLOSED_FLAG) {
    return bytes_zero((const uint8_t *)&state->group,
                      sizeof(*state) - offsetof(sagr_ccl_v1_credit_state_t,
                                                group))
               ? SAGR_CCL_V1_STATUS_CLOSED
               : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (state->flags != 0U || state->reserved0 != 0U ||
      !bytes_zero(state->reserved, sizeof(state->reserved)) ||
      sagr_ccl_v1_group_identity_validate(&state->group) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      state->self_rank >= state->group.world_size ||
      state->credits_per_peer == 0U ||
      state->credits_per_peer > SAGR_CCL_V1_MAX_CREDITS_PER_PEER) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  mask = credit_mask(state->credits_per_peer);
  for (rank = 0U; rank < SAGR_CCL_V1_MAX_WORLD_SIZE; ++rank) {
    uint32_t slot;
    if (rank >= state->group.world_size || rank == state->self_rank) {
      if (state->occupied_mask[rank] != 0U ||
          state->next_generation[rank] != 0U) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
    } else if ((state->occupied_mask[rank] & ~mask) != 0U ||
               state->next_generation[rank] == 0U ||
               state->next_generation[rank] == UINT64_MAX) {
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
    for (slot = 0U; slot < SAGR_CCL_V1_MAX_CREDITS_PER_PEER; ++slot) {
      const int occupied = (state->occupied_mask[rank] &
                            (UINT32_C(1) << slot)) != 0U;
      if ((occupied && state->slot_generation[rank][slot] == 0U) ||
          (!occupied && state->slot_generation[rank][slot] != 0U)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
    }
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_credit_state_init(
    sagr_ccl_v1_credit_state_t *state, uint32_t state_size,
    const sagr_ccl_v1_group_identity_t *group, uint32_t self_rank,
    uint32_t credits_per_peer) {
  uint32_t rank;
  if (state == NULL || group == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (state_size < sizeof(*state)) {
    if (state_size >= sizeof(state->struct_size)) {
      state->struct_size = (uint32_t)sizeof(*state);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  if (sagr_ccl_v1_group_identity_validate(group) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      self_rank >= group->world_size || credits_per_peer == 0U ||
      credits_per_peer > SAGR_CCL_V1_MAX_CREDITS_PER_PEER) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  state->struct_size = (uint32_t)sizeof(*state);
  state->group = *group;
  state->self_rank = self_rank;
  state->credits_per_peer = credits_per_peer;
  for (rank = 0U; rank < group->world_size; ++rank) {
    if (rank != self_rank) {
      state->next_generation[rank] = 1U;
    }
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_credit_acquire(
    sagr_ccl_v1_credit_state_t *state, uint32_t destination_rank,
    uint32_t *slot_index, uint64_t *slot_generation) {
  sagr_ccl_v1_status_t status = validate_credit_state(state);
  uint32_t slot;
  if (slot_index == NULL || slot_generation == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  *slot_index = UINT32_MAX;
  *slot_generation = 0U;
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (destination_rank >= state->group.world_size ||
      destination_rank == state->self_rank) {
    return SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH;
  }
  for (slot = 0U; slot < state->credits_per_peer; ++slot) {
    const uint32_t bit = UINT32_C(1) << slot;
    if ((state->occupied_mask[destination_rank] & bit) == 0U) {
      const uint64_t generation = state->next_generation[destination_rank];
      if (generation == UINT64_MAX - UINT64_C(1)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      state->occupied_mask[destination_rank] |= bit;
      state->slot_generation[destination_rank][slot] = generation;
      state->next_generation[destination_rank] = generation + UINT64_C(1);
      *slot_index = slot;
      *slot_generation = generation;
      return SAGR_CCL_V1_STATUS_SUCCESS;
    }
  }
  return SAGR_CCL_V1_STATUS_BUSY;
}

sagr_ccl_v1_status_t sagr_ccl_v1_credit_release(
    sagr_ccl_v1_credit_state_t *state, uint32_t destination_rank,
    uint32_t slot_index, uint64_t slot_generation) {
  sagr_ccl_v1_status_t status = validate_credit_state(state);
  uint32_t bit;
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (destination_rank >= state->group.world_size ||
      destination_rank == state->self_rank ||
      slot_index >= state->credits_per_peer || slot_generation == 0U) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  bit = UINT32_C(1) << slot_index;
  if ((state->occupied_mask[destination_rank] & bit) == 0U) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  if (state->slot_generation[destination_rank][slot_index] !=
      slot_generation) {
    return SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH;
  }
  state->occupied_mask[destination_rank] &= ~bit;
  state->slot_generation[destination_rank][slot_index] = 0U;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_credit_state_close(
    sagr_ccl_v1_credit_state_t *state) {
  uint32_t rank;
  sagr_ccl_v1_status_t status = validate_credit_state(state);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  for (rank = 0U; rank < state->group.world_size; ++rank) {
    if (state->occupied_mask[rank] != 0U) {
      return SAGR_CCL_V1_STATUS_BUSY;
    }
  }
  memset(state, 0, sizeof(*state));
  state->struct_size = (uint32_t)sizeof(*state);
  state->flags = CREDIT_CLOSED_FLAG;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static int record_equal(const sagr_ccl_v1_carrier_record_t *left,
                        const sagr_ccl_v1_carrier_record_t *right) {
  uint8_t left_wire[SAGR_CCL_V1_CARRIER_WIRE_BYTES];
  uint8_t right_wire[SAGR_CCL_V1_CARRIER_WIRE_BYTES];
  return sagr_ccl_v1_carrier_record_encode(left, left_wire,
                                           (uint32_t)sizeof(left_wire)) ==
             SAGR_CCL_V1_STATUS_SUCCESS &&
         sagr_ccl_v1_carrier_record_encode(right, right_wire,
                                            (uint32_t)sizeof(right_wire)) ==
             SAGR_CCL_V1_STATUS_SUCCESS &&
         memcmp(left_wire, right_wire, sizeof(left_wire)) == 0;
}

static sagr_ccl_v1_status_t validate_slot(
    const sagr_ccl_v1_carrier_slot_state_t *slot) {
  if (slot == NULL || slot->struct_size != sizeof(*slot)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (slot->reserved0 != 0U ||
      !bytes_zero(slot->reserved, sizeof(slot->reserved))) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  switch (slot->phase) {
    case SAGR_CCL_V1_CARRIER_SLOT_EMPTY:
      if (!bytes_zero((const uint8_t *)&slot->record, sizeof(slot->record)) ||
          slot->abort_status != SAGR_CCL_V1_STATUS_SUCCESS) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_CARRIER_SLOT_READY:
    case SAGR_CCL_V1_CARRIER_SLOT_CONSUMED:
      if (sagr_ccl_v1_carrier_record_validate(&slot->record) !=
              SAGR_CCL_V1_STATUS_SUCCESS ||
          slot->abort_status != SAGR_CCL_V1_STATUS_SUCCESS) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_CARRIER_SLOT_ABORTED:
      if (!bytes_zero((const uint8_t *)&slot->record, sizeof(slot->record)) ||
          !valid_abort_status((sagr_ccl_v1_status_t)slot->abort_status)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_CARRIER_SLOT_CLOSED:
      if (!bytes_zero((const uint8_t *)&slot->record, sizeof(slot->record)) ||
          (slot->abort_status != SAGR_CCL_V1_STATUS_SUCCESS &&
           !valid_abort_status(
               (sagr_ccl_v1_status_t)slot->abort_status))) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    default:
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_init(
    sagr_ccl_v1_carrier_slot_state_t *slot, uint32_t slot_size) {
  if (slot == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (slot_size < sizeof(*slot)) {
    if (slot_size >= sizeof(slot->struct_size)) {
      slot->struct_size = (uint32_t)sizeof(*slot);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  memset(slot, 0, sizeof(*slot));
  slot->struct_size = (uint32_t)sizeof(*slot);
  slot->phase = SAGR_CCL_V1_CARRIER_SLOT_EMPTY;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_publish(
    sagr_ccl_v1_carrier_slot_state_t *slot,
    const sagr_ccl_v1_carrier_record_t *record) {
  sagr_ccl_v1_status_t status = validate_slot(slot);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  status = sagr_ccl_v1_carrier_record_validate(record);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (slot->phase == SAGR_CCL_V1_CARRIER_SLOT_ABORTED) {
    return SAGR_CCL_V1_STATUS_ABORTED;
  }
  if (slot->phase == SAGR_CCL_V1_CARRIER_SLOT_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  if (slot->phase != SAGR_CCL_V1_CARRIER_SLOT_EMPTY) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  slot->record = *record;
  slot->phase = SAGR_CCL_V1_CARRIER_SLOT_READY;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_consume(
    sagr_ccl_v1_carrier_slot_state_t *slot,
    const sagr_ccl_v1_carrier_record_t *record) {
  sagr_ccl_v1_status_t status = validate_slot(slot);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (slot->phase == SAGR_CCL_V1_CARRIER_SLOT_ABORTED) {
    return SAGR_CCL_V1_STATUS_ABORTED;
  }
  if (slot->phase == SAGR_CCL_V1_CARRIER_SLOT_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  if (slot->phase != SAGR_CCL_V1_CARRIER_SLOT_READY) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  if (!record_equal(&slot->record, record)) {
    return SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH;
  }
  slot->phase = SAGR_CCL_V1_CARRIER_SLOT_CONSUMED;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_release(
    sagr_ccl_v1_carrier_slot_state_t *slot,
    const sagr_ccl_v1_carrier_record_t *record) {
  sagr_ccl_v1_status_t status = validate_slot(slot);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (slot->phase != SAGR_CCL_V1_CARRIER_SLOT_CONSUMED) {
    return slot->phase == SAGR_CCL_V1_CARRIER_SLOT_ABORTED
               ? SAGR_CCL_V1_STATUS_ABORTED
               : SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  if (!record_equal(&slot->record, record)) {
    return SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH;
  }
  memset(&slot->record, 0, sizeof(slot->record));
  slot->phase = SAGR_CCL_V1_CARRIER_SLOT_EMPTY;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_abort(
    sagr_ccl_v1_carrier_slot_state_t *slot,
    sagr_ccl_v1_status_t reason) {
  sagr_ccl_v1_status_t status = validate_slot(slot);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (!valid_abort_status(reason)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (slot->phase == SAGR_CCL_V1_CARRIER_SLOT_ABORTED) {
    return (sagr_ccl_v1_status_t)slot->abort_status;
  }
  if (slot->phase == SAGR_CCL_V1_CARRIER_SLOT_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  memset(&slot->record, 0, sizeof(slot->record));
  slot->phase = SAGR_CCL_V1_CARRIER_SLOT_ABORTED;
  slot->abort_status = (uint32_t)reason;
  return reason;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_slot_close(
    sagr_ccl_v1_carrier_slot_state_t *slot) {
  sagr_ccl_v1_status_t status = validate_slot(slot);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (slot->phase == SAGR_CCL_V1_CARRIER_SLOT_READY ||
      slot->phase == SAGR_CCL_V1_CARRIER_SLOT_CONSUMED) {
    return SAGR_CCL_V1_STATUS_BUSY;
  }
  if (slot->phase == SAGR_CCL_V1_CARRIER_SLOT_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  slot->phase = SAGR_CCL_V1_CARRIER_SLOT_CLOSED;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static int write_all(int descriptor, const uint8_t *bytes, size_t count) {
  size_t offset = 0U;
  while (offset < count) {
    const ssize_t written = pwrite(descriptor, bytes + offset, count - offset,
                                   (off_t)offset);
    if (written > 0) {
      offset += (size_t)written;
    } else if (written < 0 && errno == EINTR) {
      continue;
    } else {
      return -1;
    }
  }
  return 0;
}

static int read_all(int descriptor, uint8_t *bytes, size_t count) {
  size_t offset = 0U;
  while (offset < count) {
    const ssize_t got = pread(descriptor, bytes + offset, count - offset,
                              (off_t)offset);
    if (got > 0) {
      offset += (size_t)got;
    } else if (got < 0 && errno == EINTR) {
      continue;
    } else {
      return -1;
    }
  }
  return 0;
}

static sagr_ccl_v1_status_t validate_payload_descriptor(
    int descriptor, uint64_t byte_count) {
  const int required_seals =
      F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL;
  struct stat attributes;
  int descriptor_flags;
  int status_flags;
  int seals;
  if (descriptor < 0 || byte_count == 0U ||
      byte_count > SAGR_CCL_V1_CARRIER_MAX_PAYLOAD_BYTES ||
      byte_count > (uint64_t)SIZE_MAX || byte_count > (uint64_t)INT64_MAX) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  descriptor_flags = fcntl(descriptor, F_GETFD);
  status_flags = fcntl(descriptor, F_GETFL);
  seals = fcntl(descriptor, F_GET_SEALS);
  if (descriptor_flags < 0 || status_flags < 0 || seals < 0 ||
      fstat(descriptor, &attributes) != 0) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (!S_ISREG(attributes.st_mode) || attributes.st_nlink != 0 ||
      attributes.st_uid != geteuid() ||
      (attributes.st_mode & (mode_t)07777) != (mode_t)0600 ||
      attributes.st_size < 0 ||
      (uint64_t)attributes.st_size != byte_count ||
      (descriptor_flags & FD_CLOEXEC) == 0 ||
      (status_flags & O_ACCMODE) != O_RDONLY || seals != required_seals) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static sagr_ccl_v1_status_t payload_crc_from_fd(
    int descriptor, uint64_t byte_count, uint32_t *payload_crc32c) {
  uint8_t scratch[65536];
  uint32_t crc = UINT32_MAX;
  uint64_t offset = 0U;
  if (payload_crc32c == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  while (offset < byte_count) {
    const uint64_t remaining = byte_count - offset;
    const size_t count = remaining < sizeof(scratch) ? (size_t)remaining
                                                     : sizeof(scratch);
    size_t collected = 0U;
    while (collected < count) {
      const ssize_t got = pread(descriptor, scratch + collected,
                                count - collected,
                                (off_t)(offset + collected));
      if (got > 0) {
        collected += (size_t)got;
      } else if (got < 0 && errno == EINTR) {
        continue;
      } else {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
    }
    crc = crc32c_extend(crc, scratch, count);
    offset += count;
  }
  *payload_crc32c = ~crc;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_payload_create(
    const void *bytes, uint64_t byte_count, int *descriptor,
    uint32_t *payload_crc32c) {
  const int seals = F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL;
  char path[64];
  int writable = -1;
  int readonly = -1;
  int saved_errno = 0;
  int length;
  if (descriptor == NULL || payload_crc32c == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  *descriptor = -1;
  *payload_crc32c = 0U;
  if (bytes == NULL || byte_count == 0U ||
      byte_count > SAGR_CCL_V1_CARRIER_MAX_PAYLOAD_BYTES ||
      byte_count > (uint64_t)SIZE_MAX || byte_count > (uint64_t)INT64_MAX) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  writable = memfd_create("sagr-ccl-v1", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (writable < 0 || fchmod(writable, (mode_t)0600) != 0 ||
      ftruncate(writable, (off_t)byte_count) != 0 ||
      write_all(writable, (const uint8_t *)bytes, (size_t)byte_count) != 0 ||
      fcntl(writable, F_ADD_SEALS, seals) != 0) {
    saved_errno = errno;
    if (writable >= 0) {
      (void)close(writable);
    }
    return status_from_resource_errno(saved_errno);
  }
  length = snprintf(path, sizeof(path), "/proc/self/fd/%d", writable);
  if (length <= 0 || (size_t)length >= sizeof(path)) {
    (void)close(writable);
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  readonly = open(path, O_RDONLY | O_CLOEXEC);
  saved_errno = errno;
  (void)close(writable);
  if (readonly < 0) {
    return status_from_resource_errno(saved_errno);
  }
  if (validate_payload_descriptor(readonly, byte_count) !=
      SAGR_CCL_V1_STATUS_SUCCESS) {
    if (readonly >= 0) {
      (void)close(readonly);
    }
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (payload_crc_from_fd(readonly, byte_count, payload_crc32c) !=
      SAGR_CCL_V1_STATUS_SUCCESS) {
    (void)close(readonly);
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  *descriptor = readonly;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_payload_validate(
    int descriptor, uint64_t byte_count, uint32_t payload_crc32c) {
  uint32_t observed_crc = 0U;
  sagr_ccl_v1_status_t status =
      validate_payload_descriptor(descriptor, byte_count);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  status = payload_crc_from_fd(descriptor, byte_count, &observed_crc);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  return observed_crc == payload_crc32c ? SAGR_CCL_V1_STATUS_SUCCESS
                                        : SAGR_CCL_V1_STATUS_CHECKSUM_ERROR;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_payload_copy(
    int descriptor, uint64_t byte_count, uint32_t payload_crc32c,
    void *destination, uint64_t destination_capacity) {
  sagr_ccl_v1_status_t status;
  if (destination == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (destination_capacity < byte_count) {
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  status = sagr_ccl_v1_carrier_payload_validate(descriptor, byte_count,
                                                payload_crc32c);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  return read_all(descriptor, (uint8_t *)destination,
                  (size_t)byte_count) == 0
             ? SAGR_CCL_V1_STATUS_SUCCESS
             : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
}

static sagr_ccl_v1_status_t validate_seqpacket(int descriptor) {
  int type = 0;
  int domain = 0;
  int descriptor_flags;
  int status_flags;
  struct ucred credentials;
  socklen_t size = (socklen_t)sizeof(type);
  if (descriptor < 0 ||
      getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &type, &size) != 0 ||
      size != sizeof(type) || type != SOCK_SEQPACKET) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  size = (socklen_t)sizeof(domain);
  if (getsockopt(descriptor, SOL_SOCKET, SO_DOMAIN, &domain, &size) != 0 ||
      size != sizeof(domain) || domain != AF_UNIX) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  descriptor_flags = fcntl(descriptor, F_GETFD);
  status_flags = fcntl(descriptor, F_GETFL);
  size = (socklen_t)sizeof(credentials);
  if (descriptor_flags < 0 || status_flags < 0 ||
      (descriptor_flags & FD_CLOEXEC) == 0 ||
      (status_flags & O_NONBLOCK) == 0 ||
      getsockopt(descriptor, SOL_SOCKET, SO_PEERCRED, &credentials, &size) !=
          0 ||
      size != sizeof(credentials) || credentials.uid != geteuid()) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_send(
    int socket_descriptor, const sagr_ccl_v1_carrier_record_t *record,
    int payload_descriptor) {
  uint8_t wire[SAGR_CCL_V1_CARRIER_WIRE_BYTES];
  struct iovec vector;
  struct msghdr message;
  struct cmsghdr *control_message;
  unsigned char control[CMSG_SPACE(sizeof(int))];
  ssize_t sent;
  sagr_ccl_v1_status_t status = validate_seqpacket(socket_descriptor);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  status = sagr_ccl_v1_carrier_record_encode(record, wire,
                                             (uint32_t)sizeof(wire));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if ((record->payload_bytes == 0U && payload_descriptor >= 0) ||
      (record->payload_bytes != 0U &&
       validate_payload_descriptor(payload_descriptor,
                                   record->payload_bytes) !=
           SAGR_CCL_V1_STATUS_SUCCESS)) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  memset(&message, 0, sizeof(message));
  memset(control, 0, sizeof(control));
  vector.iov_base = wire;
  vector.iov_len = sizeof(wire);
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  if (record->payload_bytes != 0U) {
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    control_message = CMSG_FIRSTHDR(&message);
    if (control_message == NULL) {
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
    control_message->cmsg_level = SOL_SOCKET;
    control_message->cmsg_type = SCM_RIGHTS;
    control_message->cmsg_len = CMSG_LEN(sizeof(payload_descriptor));
    memcpy(CMSG_DATA(control_message), &payload_descriptor,
           sizeof(payload_descriptor));
  }
  do {
    sent = sendmsg(socket_descriptor, &message, MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  if (sent == (ssize_t)sizeof(wire)) {
    return SAGR_CCL_V1_STATUS_SUCCESS;
  }
  if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return SAGR_CCL_V1_STATUS_BUSY;
  }
  if (sent < 0 &&
      (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)) {
    return SAGR_CCL_V1_STATUS_PEER_LOST;
  }
  return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
}

static size_t collect_descriptors(struct msghdr *message,
                                  int descriptors[4], int *malformed) {
  struct cmsghdr *control_message;
  size_t count = 0U;
  *malformed = 0;
  for (control_message = CMSG_FIRSTHDR(message); control_message != NULL;
       control_message = CMSG_NXTHDR(message, control_message)) {
    if (control_message->cmsg_level != SOL_SOCKET ||
        control_message->cmsg_type != SCM_RIGHTS ||
        control_message->cmsg_len < CMSG_LEN(0)) {
      *malformed = 1;
      continue;
    }
    {
      const size_t bytes = control_message->cmsg_len - CMSG_LEN(0);
      const size_t received = bytes / sizeof(int);
      const int *values = (const int *)CMSG_DATA(control_message);
      size_t index;
      if (bytes == 0U || bytes % sizeof(int) != 0U) {
        *malformed = 1;
      }
      for (index = 0U; index < received; ++index) {
        if (count < 4U) {
          descriptors[count] = values[index];
        } else {
          (void)close(values[index]);
        }
        ++count;
      }
    }
  }
  if (count > 4U) {
    *malformed = 1;
  }
  return count;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_receive(
    int socket_descriptor, sagr_ccl_v1_carrier_record_t *record,
    uint32_t record_size, int *payload_descriptor) {
  uint8_t wire[SAGR_CCL_V1_CARRIER_WIRE_BYTES];
  struct iovec vector;
  struct msghdr message;
  unsigned char control[CMSG_SPACE(sizeof(int) * 4U)];
  int descriptors[4] = {-1, -1, -1, -1};
  size_t descriptor_count;
  size_t index;
  int malformed_control = 0;
  sagr_ccl_v1_carrier_record_t decoded;
  ssize_t received;
  sagr_ccl_v1_status_t status = validate_seqpacket(socket_descriptor);
  if (record == NULL || payload_descriptor == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (record_size < sizeof(*record)) {
    if (record_size >= sizeof(record->struct_size)) {
      record->struct_size = (uint32_t)sizeof(*record);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  memset(&message, 0, sizeof(message));
  memset(control, 0, sizeof(control));
  vector.iov_base = wire;
  vector.iov_len = sizeof(wire);
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  do {
    received = recvmsg(socket_descriptor, &message, MSG_CMSG_CLOEXEC);
  } while (received < 0 && errno == EINTR);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return SAGR_CCL_V1_STATUS_BUSY;
  }
  if (received < 0 &&
      (errno == ECONNRESET || errno == ENOTCONN || errno == EPIPE)) {
    return SAGR_CCL_V1_STATUS_PEER_LOST;
  }
  if (received < 0) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  descriptor_count =
      collect_descriptors(&message, descriptors, &malformed_control);
  if (received == 0) {
    status = descriptor_count == 0U && malformed_control == 0 &&
                     (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) == 0
                 ? SAGR_CCL_V1_STATUS_PEER_LOST
                 : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    goto fail;
  }
  if (received != (ssize_t)sizeof(wire) ||
      (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0 ||
      malformed_control != 0) {
    status = SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    goto fail;
  }
  status = sagr_ccl_v1_carrier_record_decode(
      wire, (uint32_t)sizeof(wire), &decoded, (uint32_t)sizeof(decoded));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    goto fail;
  }
  if ((decoded.payload_bytes == 0U && descriptor_count != 0U) ||
      (decoded.payload_bytes != 0U && descriptor_count != 1U)) {
    status = SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    goto fail;
  }
  if (descriptor_count == 1U) {
    status = sagr_ccl_v1_carrier_payload_validate(
        descriptors[0], decoded.payload_bytes, decoded.payload_crc32c);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      goto fail;
    }
  }
  *record = decoded;
  *payload_descriptor = descriptor_count == 1U ? descriptors[0] : -1;
  return SAGR_CCL_V1_STATUS_SUCCESS;

fail:
  for (index = 0U; index < descriptor_count && index < 4U; ++index) {
    if (descriptors[index] >= 0) {
      (void)close(descriptors[index]);
    }
  }
  return status;
}

static void owned_transfer_clear(carrier_owned_transfer_t *transfer) {
  if (transfer->descriptor >= 0) {
    (void)close(transfer->descriptor);
  }
  memset(transfer, 0, sizeof(*transfer));
  transfer->descriptor = -1;
}

static int transfer_tuple_equal(
    const sagr_ccl_v1_carrier_record_t *left,
    const sagr_ccl_v1_carrier_record_t *right) {
  return sagr_ccl_v1_group_identity_equal(&left->group, &right->group) &&
         memcmp(left->descriptor_sha256, right->descriptor_sha256,
                sizeof(left->descriptor_sha256)) == 0 &&
         left->sequence == right->sequence &&
         left->slot_generation == right->slot_generation &&
         left->phase == right->phase &&
         left->step_index == right->step_index &&
         left->chunk_index == right->chunk_index &&
         left->source_rank == right->source_rank &&
         left->destination_rank == right->destination_rank &&
         left->slot_index == right->slot_index;
}

static sagr_ccl_v1_status_t validate_session(
    const sagr_ccl_v1_carrier_session_t session) {
  uint32_t rank;
  uint32_t slot;
  if (session == NULL || session->magic != CARRIER_SESSION_MAGIC ||
      sagr_ccl_v1_group_identity_validate(&session->group) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      session->self_rank >= session->group.world_size ||
      session->credits_per_peer == 0U ||
      session->credits_per_peer > SAGR_CCL_V1_MAX_CREDITS_PER_PEER) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (session->phase == SAGR_CCL_V1_CARRIER_SESSION_RUNNING) {
    if (session->first_error != SAGR_CCL_V1_STATUS_SUCCESS ||
        session->failed_rank != SAGR_CCL_V1_NO_RANK ||
        bytes_nonzero((const uint8_t *)&session->abort_record,
                      sizeof(session->abort_record)) ||
        validate_credit_state(&session->credits) !=
            SAGR_CCL_V1_STATUS_SUCCESS) {
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
  } else if (session->phase == SAGR_CCL_V1_CARRIER_SESSION_ABORTED) {
    if (!valid_abort_status(session->first_error) ||
        (session->failed_rank != SAGR_CCL_V1_NO_RANK &&
         session->failed_rank >= session->group.world_size)) {
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
    if (!bytes_nonzero((const uint8_t *)&session->abort_record,
                       sizeof(session->abort_record)) ||
        sagr_ccl_v1_carrier_record_validate(&session->abort_record) !=
             SAGR_CCL_V1_STATUS_SUCCESS ||
         session->abort_record.kind != SAGR_CCL_V1_CARRIER_MESSAGE_ABORT ||
         session->abort_record.status != session->first_error ||
         session->abort_record.failed_rank != session->failed_rank ||
         !sagr_ccl_v1_group_identity_equal(&session->abort_record.group,
                                           &session->group)) {
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
  } else if (session->phase != SAGR_CCL_V1_CARRIER_SESSION_CLOSED) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  for (rank = 0U; rank < SAGR_CCL_V1_MAX_WORLD_SIZE; ++rank) {
    for (slot = 0U; slot < SAGR_CCL_V1_MAX_CREDITS_PER_PEER; ++slot) {
      const carrier_owned_transfer_t *sender = &session->sender[rank][slot];
      const carrier_owned_transfer_t *receiver =
          &session->receiver[rank][slot];
      const uint32_t credit_bit = UINT32_C(1) << slot;
      if (session->phase != SAGR_CCL_V1_CARRIER_SESSION_RUNNING ||
          rank >= session->group.world_size || rank == session->self_rank ||
          slot >= session->credits_per_peer) {
        if (sender->phase != CARRIER_OWNED_EMPTY || sender->descriptor != -1 ||
            receiver->phase != CARRIER_OWNED_EMPTY ||
            receiver->descriptor != -1) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
        if (session->receiver_last_generation[rank][slot] != 0U) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
        continue;
      }
      if (sender->phase == CARRIER_OWNED_EMPTY) {
        if (sender->descriptor != -1 ||
            (session->credits.occupied_mask[rank] & credit_bit) != 0U ||
            session->credits.slot_generation[rank][slot] != 0U) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
      } else if (sender->phase == CARRIER_OWNED_PREPARED) {
        if (sagr_ccl_v1_carrier_record_validate(&sender->record) !=
                SAGR_CCL_V1_STATUS_SUCCESS ||
            sender->record.kind != SAGR_CCL_V1_CARRIER_MESSAGE_DATA ||
            sender->record.source_rank != session->self_rank ||
            sender->record.destination_rank != rank ||
            sender->record.slot_index != slot ||
            ((sender->record.payload_bytes == 0U && sender->descriptor != -1) ||
             (sender->record.payload_bytes != 0U &&
              sender->descriptor < 0))) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
        if ((session->credits.occupied_mask[rank] & credit_bit) == 0U ||
            session->credits.slot_generation[rank][slot] !=
                sender->record.slot_generation) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
      } else if (sender->phase == CARRIER_OWNED_SENT) {
        if (sender->descriptor != -1 ||
            sagr_ccl_v1_carrier_record_validate(&sender->record) !=
                SAGR_CCL_V1_STATUS_SUCCESS ||
            sender->record.kind != SAGR_CCL_V1_CARRIER_MESSAGE_DATA ||
            sender->record.source_rank != session->self_rank ||
            sender->record.destination_rank != rank ||
            sender->record.slot_index != slot) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
        if ((session->credits.occupied_mask[rank] & credit_bit) == 0U ||
            session->credits.slot_generation[rank][slot] !=
                sender->record.slot_generation) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
      } else {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      if (receiver->phase == CARRIER_OWNED_EMPTY) {
        if (receiver->descriptor != -1) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
      } else if (receiver->phase == CARRIER_OWNED_READY) {
        if (sagr_ccl_v1_carrier_record_validate(&receiver->record) !=
                SAGR_CCL_V1_STATUS_SUCCESS ||
            receiver->record.kind != SAGR_CCL_V1_CARRIER_MESSAGE_DATA ||
            receiver->record.source_rank != rank ||
            receiver->record.destination_rank != session->self_rank ||
            receiver->record.slot_index != slot ||
            ((receiver->record.payload_bytes == 0U &&
              receiver->descriptor != -1) ||
             (receiver->record.payload_bytes != 0U &&
              receiver->descriptor < 0))) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
        if (session->receiver_last_generation[rank][slot] !=
            receiver->record.slot_generation) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
      } else if (receiver->phase == CARRIER_OWNED_CONSUMED) {
        if (receiver->descriptor != -1 ||
            sagr_ccl_v1_carrier_record_validate(&receiver->record) !=
                SAGR_CCL_V1_STATUS_SUCCESS ||
            receiver->record.kind != SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED ||
            receiver->record.source_rank != rank ||
            receiver->record.destination_rank != session->self_rank ||
            receiver->record.slot_index != slot) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
        if (session->receiver_last_generation[rank][slot] !=
            receiver->record.slot_generation) {
          return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
        }
      } else {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
    }
  }
  return session->phase == SAGR_CCL_V1_CARRIER_SESSION_ABORTED
             ? SAGR_CCL_V1_STATUS_ABORTED
             : session->phase == SAGR_CCL_V1_CARRIER_SESSION_CLOSED
                   ? SAGR_CCL_V1_STATUS_CLOSED
                   : SAGR_CCL_V1_STATUS_SUCCESS;
}

static sagr_ccl_v1_status_t abort_record_from_context(
    const sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_carrier_record_t *context,
    sagr_ccl_v1_status_t reason, uint32_t failed_rank,
    sagr_ccl_v1_carrier_record_t *abort_record) {
  sagr_ccl_v1_status_t status;
  if (context == NULL || abort_record == NULL ||
      sagr_ccl_v1_carrier_record_validate(context) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      !sagr_ccl_v1_group_identity_equal(&context->group, &session->group)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  status = sagr_ccl_v1_carrier_record_init(
      abort_record, (uint32_t)sizeof(*abort_record));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  abort_record->group = session->group;
  memcpy(abort_record->descriptor_sha256, context->descriptor_sha256,
         sizeof(abort_record->descriptor_sha256));
  abort_record->sequence = context->sequence;
  abort_record->kind = SAGR_CCL_V1_CARRIER_MESSAGE_ABORT;
  abort_record->phase = SAGR_CCL_V1_PLAN_PHASE_INVALID;
  abort_record->step_index = SAGR_CCL_V1_NO_CHUNK;
  abort_record->chunk_index = SAGR_CCL_V1_NO_CHUNK;
  abort_record->source_rank = session->self_rank;
  abort_record->destination_rank = SAGR_CCL_V1_NO_RANK;
  abort_record->slot_index = SAGR_CCL_V1_NO_CHUNK;
  abort_record->status = valid_abort_status(reason)
                             ? reason
                             : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  abort_record->failed_rank = failed_rank < session->group.world_size
                                  ? failed_rank
                                  : SAGR_CCL_V1_NO_RANK;
  return sagr_ccl_v1_carrier_record_validate(abort_record);
}

static void session_abort_internal(
    sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_carrier_record_t *abort_record,
    sagr_ccl_v1_status_t reason, uint32_t failed_rank) {
  uint32_t rank;
  uint32_t slot;
  if (session->phase == SAGR_CCL_V1_CARRIER_SESSION_ABORTED ||
      session->phase == SAGR_CCL_V1_CARRIER_SESSION_CLOSED) {
    return;
  }
  session->phase = SAGR_CCL_V1_CARRIER_SESSION_ABORTED;
  session->first_error = valid_abort_status(reason)
                             ? reason
                             : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  session->failed_rank = failed_rank < session->group.world_size
                             ? failed_rank
                             : SAGR_CCL_V1_NO_RANK;
  session->abort_record = *abort_record;
  for (rank = 0U; rank < SAGR_CCL_V1_MAX_WORLD_SIZE; ++rank) {
    for (slot = 0U; slot < SAGR_CCL_V1_MAX_CREDITS_PER_PEER; ++slot) {
      owned_transfer_clear(&session->sender[rank][slot]);
      owned_transfer_clear(&session->receiver[rank][slot]);
    }
  }
  memset(&session->credits, 0, sizeof(session->credits));
  memset(session->receiver_last_generation, 0,
         sizeof(session->receiver_last_generation));
}

static void session_abort_from_context(
    sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_carrier_record_t *context,
    sagr_ccl_v1_status_t reason, uint32_t failed_rank) {
  sagr_ccl_v1_carrier_record_t abort_record;
  sagr_ccl_v1_status_t status = abort_record_from_context(
      session, context, reason, failed_rank, &abort_record);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return;
  }
  session_abort_internal(session, &abort_record, reason, failed_rank);
}

static void session_abort_from_descriptor(
    sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_descriptor_t *descriptor,
    sagr_ccl_v1_status_t reason, uint32_t failed_rank) {
  sagr_ccl_v1_carrier_record_t abort_record;
  sagr_ccl_v1_status_t status = sagr_ccl_v1_carrier_abort_record(
      descriptor, session->self_rank, failed_rank,
      valid_abort_status(reason) ? reason : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR,
      &abort_record, (uint32_t)sizeof(abort_record));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return;
  }
  session_abort_internal(session, &abort_record, reason, failed_rank);
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_create(
    const sagr_ccl_v1_group_identity_t *group, uint32_t self_rank,
    uint32_t credits_per_peer, sagr_ccl_v1_carrier_session_t *session) {
  sagr_ccl_v1_carrier_session_t created;
  uint32_t rank;
  uint32_t slot;
  sagr_ccl_v1_status_t status;
  if (session == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  *session = NULL;
  if (group == NULL ||
      sagr_ccl_v1_group_identity_validate(group) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      self_rank >= group->world_size || credits_per_peer == 0U ||
      credits_per_peer > SAGR_CCL_V1_MAX_CREDITS_PER_PEER) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  created = (sagr_ccl_v1_carrier_session_t)calloc(1U, sizeof(*created));
  if (created == NULL) {
    return SAGR_CCL_V1_STATUS_OUT_OF_RESOURCES;
  }
  created->magic = CARRIER_SESSION_MAGIC;
  created->phase = SAGR_CCL_V1_CARRIER_SESSION_RUNNING;
  created->self_rank = self_rank;
  created->credits_per_peer = credits_per_peer;
  created->failed_rank = SAGR_CCL_V1_NO_RANK;
  created->group = *group;
  for (rank = 0U; rank < SAGR_CCL_V1_MAX_WORLD_SIZE; ++rank) {
    for (slot = 0U; slot < SAGR_CCL_V1_MAX_CREDITS_PER_PEER; ++slot) {
      created->sender[rank][slot].descriptor = -1;
      created->receiver[rank][slot].descriptor = -1;
    }
  }
  status = sagr_ccl_v1_credit_state_init(
      &created->credits, (uint32_t)sizeof(created->credits), group, self_rank,
      credits_per_peer);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    free(created);
    return status;
  }
  *session = created;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_prepare_data(
    sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t step_index,
    const void *payload, uint64_t payload_bytes,
    sagr_ccl_v1_carrier_record_t *record, uint32_t record_size) {
  sagr_ccl_v1_carrier_record_t prepared;
  carrier_owned_transfer_t *owned;
  uint32_t slot = 0U;
  uint64_t generation = 0U;
  uint32_t payload_crc = 0U;
  int payload_descriptor = -1;
  sagr_ccl_v1_status_t status;
  if (record == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (record_size < sizeof(*record)) {
    if (record_size >= sizeof(record->struct_size)) {
      record->struct_size = (uint32_t)sizeof(*record);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  status = validate_session(session);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (descriptor == NULL ||
      sagr_ccl_v1_descriptor_validate(descriptor) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      descriptor->rank != session->self_rank ||
      !sagr_ccl_v1_group_identity_equal(&descriptor->group, &session->group)) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  status = sagr_ccl_v1_carrier_record_from_plan(
      descriptor, step_index, session->self_rank,
      SAGR_CCL_V1_CARRIER_MESSAGE_DATA, 0U, 1U, 0U, &prepared,
      (uint32_t)sizeof(prepared));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (payload_bytes != prepared.payload_bytes ||
      (payload_bytes == 0U ? payload != NULL : payload == NULL)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  status = sagr_ccl_v1_credit_acquire(&session->credits,
                                      prepared.destination_rank, &slot,
                                      &generation);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (payload_bytes != 0U) {
    status = sagr_ccl_v1_carrier_payload_create(
        payload, payload_bytes, &payload_descriptor, &payload_crc);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      (void)sagr_ccl_v1_credit_release(
          &session->credits, prepared.destination_rank, slot, generation);
      return status;
    }
  }
  status = sagr_ccl_v1_carrier_record_from_plan(
      descriptor, step_index, session->self_rank,
      SAGR_CCL_V1_CARRIER_MESSAGE_DATA, slot, generation, payload_crc,
      &prepared, (uint32_t)sizeof(prepared));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    if (payload_descriptor >= 0) {
      (void)close(payload_descriptor);
    }
    (void)sagr_ccl_v1_credit_release(
        &session->credits, prepared.destination_rank, slot, generation);
    return status;
  }
  owned = &session->sender[prepared.destination_rank][slot];
  if (owned->phase != CARRIER_OWNED_EMPTY || owned->descriptor != -1) {
    if (payload_descriptor >= 0) {
      (void)close(payload_descriptor);
    }
    (void)sagr_ccl_v1_credit_release(
        &session->credits, prepared.destination_rank, slot, generation);
    session_abort_from_context(session, &prepared,
                               SAGR_CCL_V1_STATUS_PROTOCOL_ERROR,
                               session->self_rank);
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  owned->phase = CARRIER_OWNED_PREPARED;
  owned->descriptor = payload_descriptor;
  owned->record = prepared;
  *record = prepared;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_send_data(
    sagr_ccl_v1_carrier_session_t session, int socket_descriptor,
    const sagr_ccl_v1_carrier_record_t *record) {
  carrier_owned_transfer_t *owned;
  sagr_ccl_v1_status_t status = validate_session(session);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (record == NULL || record->kind != SAGR_CCL_V1_CARRIER_MESSAGE_DATA ||
      record->source_rank != session->self_rank ||
      record->destination_rank >= session->group.world_size ||
      record->slot_index >= session->credits_per_peer) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  owned = &session->sender[record->destination_rank][record->slot_index];
  if (owned->phase != CARRIER_OWNED_PREPARED ||
      !record_equal(&owned->record, record)) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  status = sagr_ccl_v1_carrier_send(socket_descriptor, record,
                                    owned->descriptor);
  if (status == SAGR_CCL_V1_STATUS_BUSY) {
    return status;
  }
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    session_abort_from_context(session, record, status,
                               record->destination_rank);
    return session->first_error;
  }
  if (owned->descriptor >= 0) {
    (void)close(owned->descriptor);
    owned->descriptor = -1;
  }
  owned->phase = CARRIER_OWNED_SENT;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_receive(
    sagr_ccl_v1_carrier_session_t session, int socket_descriptor,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t expected_step_index,
    uint32_t authenticated_peer_rank, sagr_ccl_v1_carrier_record_t *record,
    uint32_t record_size) {
  sagr_ccl_v1_carrier_record_t received;
  carrier_owned_transfer_t *owned;
  int payload_descriptor = -1;
  sagr_ccl_v1_status_t status;
  if (record == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (record_size < sizeof(*record)) {
    if (record_size >= sizeof(record->struct_size)) {
      record->struct_size = (uint32_t)sizeof(*record);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  status = validate_session(session);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (descriptor == NULL ||
      sagr_ccl_v1_descriptor_validate(descriptor) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      descriptor->rank != session->self_rank ||
      !sagr_ccl_v1_group_identity_equal(&descriptor->group, &session->group) ||
      authenticated_peer_rank >= session->group.world_size ||
      authenticated_peer_rank == session->self_rank) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  status = sagr_ccl_v1_carrier_receive(
      socket_descriptor, &received, (uint32_t)sizeof(received),
      &payload_descriptor);
  if (status == SAGR_CCL_V1_STATUS_BUSY) {
    return status;
  }
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    session_abort_from_descriptor(session, descriptor, status,
                                  authenticated_peer_rank);
    return session->first_error;
  }
  status = sagr_ccl_v1_carrier_record_validate_descriptor(
      &received, descriptor,
      received.kind == SAGR_CCL_V1_CARRIER_MESSAGE_ABORT
          ? SAGR_CCL_V1_NO_CHUNK
          : expected_step_index,
      authenticated_peer_rank);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    if (payload_descriptor >= 0) {
      (void)close(payload_descriptor);
    }
    session_abort_from_descriptor(session, descriptor, status,
                                  authenticated_peer_rank);
    return session->first_error;
  }
  if (received.kind == SAGR_CCL_V1_CARRIER_MESSAGE_ABORT) {
    session_abort_internal(session, &received, received.status,
                           received.failed_rank);
    *record = received;
    return session->first_error;
  }
  if (received.slot_index >= session->credits_per_peer) {
    if (payload_descriptor >= 0) {
      (void)close(payload_descriptor);
    }
    session_abort_from_context(session, &received,
                               SAGR_CCL_V1_STATUS_PROTOCOL_ERROR,
                               authenticated_peer_rank);
    return session->first_error;
  }
  if (received.kind == SAGR_CCL_V1_CARRIER_MESSAGE_DATA) {
    owned = &session->receiver[received.source_rank][received.slot_index];
    if (owned->phase != CARRIER_OWNED_EMPTY || owned->descriptor != -1) {
      if (payload_descriptor >= 0) {
        (void)close(payload_descriptor);
      }
      session_abort_from_context(session, &received,
                                 SAGR_CCL_V1_STATUS_OUT_OF_ORDER,
                                 authenticated_peer_rank);
      return session->first_error;
    }
    if (received.slot_generation <=
        session->receiver_last_generation[received.source_rank]
                                                 [received.slot_index]) {
      if (payload_descriptor >= 0) {
        (void)close(payload_descriptor);
      }
      session_abort_from_context(session, &received,
                                 SAGR_CCL_V1_STATUS_OUT_OF_ORDER,
                                 authenticated_peer_rank);
      return session->first_error;
    }
    session->receiver_last_generation[received.source_rank]
                                     [received.slot_index] =
        received.slot_generation;
    owned->phase = CARRIER_OWNED_READY;
    owned->descriptor = payload_descriptor;
    owned->record = received;
  } else {
    owned = &session->sender[received.destination_rank][received.slot_index];
    if (owned->phase != CARRIER_OWNED_SENT ||
        !transfer_tuple_equal(&owned->record, &received)) {
      session_abort_from_context(session, &received,
                                 SAGR_CCL_V1_STATUS_OUT_OF_ORDER,
                                 authenticated_peer_rank);
      return session->first_error;
    }
    status = sagr_ccl_v1_credit_release(
        &session->credits, received.destination_rank, received.slot_index,
        received.slot_generation);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      session_abort_from_context(session, &received, status,
                                 authenticated_peer_rank);
      return session->first_error;
    }
    owned_transfer_clear(owned);
  }
  *record = received;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_consume(
    sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t expected_step_index,
    const sagr_ccl_v1_carrier_record_t *data_record, void *destination,
    uint64_t destination_capacity,
    sagr_ccl_v1_carrier_record_t *consumed_record,
    uint32_t consumed_record_size) {
  carrier_owned_transfer_t *owned;
  sagr_ccl_v1_carrier_record_t consumed;
  sagr_ccl_v1_status_t status;
  if (data_record == NULL || consumed_record == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (consumed_record_size < sizeof(*consumed_record)) {
    if (consumed_record_size >= sizeof(consumed_record->struct_size)) {
      consumed_record->struct_size = (uint32_t)sizeof(*consumed_record);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  status = validate_session(session);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (descriptor == NULL || descriptor->rank != session->self_rank ||
      data_record->kind != SAGR_CCL_V1_CARRIER_MESSAGE_DATA ||
      data_record->destination_rank != session->self_rank ||
      data_record->source_rank >= session->group.world_size ||
      data_record->slot_index >= session->credits_per_peer ||
      (data_record->payload_bytes == 0U
           ? destination != NULL || destination_capacity != 0U
           : destination == NULL)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  status = sagr_ccl_v1_carrier_record_validate_descriptor(
      data_record, descriptor, expected_step_index, data_record->source_rank);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  owned = &session->receiver[data_record->source_rank]
                             [data_record->slot_index];
  if (owned->phase != CARRIER_OWNED_READY ||
      !record_equal(&owned->record, data_record)) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  if (data_record->payload_bytes != 0U) {
    status = sagr_ccl_v1_carrier_payload_copy(
        owned->descriptor, data_record->payload_bytes,
        data_record->payload_crc32c, destination, destination_capacity);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      if (status != SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL) {
        session_abort_from_context(session, data_record, status,
                                   data_record->source_rank);
        return session->first_error;
      }
      return status;
    }
  }
  status = sagr_ccl_v1_carrier_record_from_plan(
      descriptor, expected_step_index, data_record->source_rank,
      SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED, data_record->slot_index,
      data_record->slot_generation, 0U, &consumed,
      (uint32_t)sizeof(consumed));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS ||
      !transfer_tuple_equal(data_record, &consumed)) {
    session_abort_from_context(session, data_record,
                               SAGR_CCL_V1_STATUS_PROTOCOL_ERROR,
                               data_record->source_rank);
    return session->first_error;
  }
  if (owned->descriptor >= 0) {
    (void)close(owned->descriptor);
    owned->descriptor = -1;
  }
  owned->phase = CARRIER_OWNED_CONSUMED;
  owned->record = consumed;
  *consumed_record = consumed;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_send_consumed(
    sagr_ccl_v1_carrier_session_t session, int socket_descriptor,
    const sagr_ccl_v1_carrier_record_t *record) {
  carrier_owned_transfer_t *owned;
  sagr_ccl_v1_status_t status = validate_session(session);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (record == NULL ||
      record->kind != SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED ||
      record->destination_rank != session->self_rank ||
      record->source_rank >= session->group.world_size ||
      record->slot_index >= session->credits_per_peer) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  owned = &session->receiver[record->source_rank][record->slot_index];
  if (owned->phase != CARRIER_OWNED_CONSUMED ||
      !record_equal(&owned->record, record)) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  status = sagr_ccl_v1_carrier_send(socket_descriptor, record, -1);
  if (status == SAGR_CCL_V1_STATUS_BUSY) {
    return status;
  }
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    session_abort_from_context(session, record, status, record->source_rank);
    return session->first_error;
  }
  owned_transfer_clear(owned);
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_abort(
    sagr_ccl_v1_carrier_session_t session,
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t failed_rank,
    sagr_ccl_v1_status_t reason, sagr_ccl_v1_carrier_record_t *abort_record,
    uint32_t abort_record_size) {
  sagr_ccl_v1_carrier_record_t prepared;
  sagr_ccl_v1_status_t status;
  if (abort_record == NULL || descriptor == NULL ||
      !valid_abort_status(reason)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (abort_record_size < sizeof(*abort_record)) {
    if (abort_record_size >= sizeof(abort_record->struct_size)) {
      abort_record->struct_size = (uint32_t)sizeof(*abort_record);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  status = validate_session(session);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (descriptor->rank != session->self_rank ||
      !sagr_ccl_v1_group_identity_equal(&descriptor->group, &session->group)) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  status = sagr_ccl_v1_carrier_abort_record(
      descriptor, session->self_rank, failed_rank, reason, &prepared,
      (uint32_t)sizeof(prepared));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  session_abort_internal(session, &prepared, reason, failed_rank);
  *abort_record = prepared;
  return reason;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_send_abort(
    sagr_ccl_v1_carrier_session_t session, int socket_descriptor,
    const sagr_ccl_v1_carrier_record_t *abort_record) {
  sagr_ccl_v1_status_t status = validate_session(session);
  if (status != SAGR_CCL_V1_STATUS_ABORTED) {
    return status == SAGR_CCL_V1_STATUS_SUCCESS
               ? SAGR_CCL_V1_STATUS_OUT_OF_ORDER
               : status;
  }
  if (abort_record == NULL ||
      abort_record->kind != SAGR_CCL_V1_CARRIER_MESSAGE_ABORT ||
      abort_record->status != session->first_error ||
      abort_record->failed_rank != session->failed_rank ||
      !record_equal(abort_record, &session->abort_record)) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  return sagr_ccl_v1_carrier_send(socket_descriptor, abort_record, -1);
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_get_abort(
    sagr_ccl_v1_carrier_session_t session,
    sagr_ccl_v1_carrier_record_t *abort_record,
    uint32_t abort_record_size) {
  sagr_ccl_v1_status_t status;
  if (abort_record == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (abort_record_size < sizeof(*abort_record)) {
    if (abort_record_size >= sizeof(abort_record->struct_size)) {
      abort_record->struct_size = (uint32_t)sizeof(*abort_record);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  status = validate_session(session);
  if (status != SAGR_CCL_V1_STATUS_ABORTED) {
    return status == SAGR_CCL_V1_STATUS_SUCCESS
               ? SAGR_CCL_V1_STATUS_OUT_OF_ORDER
               : status;
  }
  *abort_record = session->abort_record;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_carrier_session_info(
    sagr_ccl_v1_carrier_session_t session,
    sagr_ccl_v1_carrier_session_info_t *info, uint32_t info_size) {
  sagr_ccl_v1_status_t status;
  uint32_t rank;
  uint32_t slot;
  if (info == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  status = validate_session(session);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS &&
      status != SAGR_CCL_V1_STATUS_ABORTED &&
      status != SAGR_CCL_V1_STATUS_CLOSED) {
    return status;
  }
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->phase = session->phase;
  info->self_rank = session->self_rank;
  info->world_size = session->group.world_size;
  info->credits_per_peer = session->credits_per_peer;
  info->first_error = session->first_error;
  info->failed_rank = session->failed_rank;
  for (rank = 0U; rank < session->group.world_size; ++rank) {
    for (slot = 0U; slot < session->credits_per_peer; ++slot) {
      if (session->sender[rank][slot].phase != CARRIER_OWNED_EMPTY) {
        ++info->sender_inflight;
      }
      if (session->receiver[rank][slot].phase == CARRIER_OWNED_READY) {
        ++info->receiver_ready;
      } else if (session->receiver[rank][slot].phase ==
                 CARRIER_OWNED_CONSUMED) {
        ++info->receiver_consumed;
      }
    }
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

void sagr_ccl_v1_carrier_session_destroy(
    sagr_ccl_v1_carrier_session_t *session) {
  sagr_ccl_v1_carrier_session_t value;
  uint32_t rank;
  uint32_t slot;
  if (session == NULL || *session == NULL) {
    return;
  }
  value = *session;
  if (value->magic != CARRIER_SESSION_MAGIC) {
    *session = NULL;
    return;
  }
  for (rank = 0U; rank < SAGR_CCL_V1_MAX_WORLD_SIZE; ++rank) {
    for (slot = 0U; slot < SAGR_CCL_V1_MAX_CREDITS_PER_PEER; ++slot) {
      owned_transfer_clear(&value->sender[rank][slot]);
      owned_transfer_clear(&value->receiver[rank][slot]);
    }
  }
  value->phase = SAGR_CCL_V1_CARRIER_SESSION_CLOSED;
  value->magic = 0U;
  free(value);
  *session = NULL;
}
