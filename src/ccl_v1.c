/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/ccl_v1.h>

#include "sha256_internal.h"

#include <stddef.h>
#include <string.h>

#define SAGR_CCL_V1_WIRE_MAGIC UINT32_C(0x53434331)
#define SAGR_CCL_V1_WIRE_CRC_OFFSET UINT32_C(144)
#define SAGR_CCL_V1_WIRE_RESERVED_OFFSET UINT32_C(148)

static int bytes_are_zero(const uint8_t *bytes, size_t count) {
  size_t index;
  for (index = 0; index < count; ++index) {
    if (bytes[index] != 0U) {
      return 0;
    }
  }
  return 1;
}

static int bytes_are_nonzero(const uint8_t *bytes, size_t count) {
  return !bytes_are_zero(bytes, count);
}

static int supported_world_size(uint32_t world_size) {
  return world_size >= SAGR_CCL_V1_MIN_WORLD_SIZE &&
         world_size <= SAGR_CCL_V1_MAX_WORLD_SIZE;
}

static uint32_t world_mask(uint32_t world_size) {
  return (UINT32_C(1) << world_size) - UINT32_C(1);
}

static int valid_copy_dtype(uint32_t dtype) {
  return dtype == SAGR_CCL_V1_DTYPE_BF16 ||
         dtype == SAGR_CCL_V1_DTYPE_FP32 ||
         dtype == SAGR_CCL_V1_DTYPE_UINT8 ||
         dtype == SAGR_CCL_V1_DTYPE_INT32 ||
         dtype == SAGR_CCL_V1_DTYPE_UINT32;
}

static int valid_sum_dtype(uint32_t dtype) {
  return dtype == SAGR_CCL_V1_DTYPE_BF16 ||
         dtype == SAGR_CCL_V1_DTYPE_FP32;
}

static int multiply_u64(uint64_t left, uint64_t right, uint64_t *result) {
  if (result == NULL || (left != 0U && right > UINT64_MAX / left)) {
    return 0;
  }
  *result = left * right;
  return 1;
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
  return (uint16_t)(((uint16_t)source[0] << 8U) | (uint16_t)source[1]);
}

static uint32_t get_u32(const uint8_t *source) {
  return ((uint32_t)source[0] << 24U) | ((uint32_t)source[1] << 16U) |
         ((uint32_t)source[2] << 8U) | (uint32_t)source[3];
}

static uint64_t get_u64(const uint8_t *source) {
  return ((uint64_t)get_u32(source) << 32U) | (uint64_t)get_u32(source + 4);
}

static uint32_t crc32c(const uint8_t *bytes, size_t count) {
  uint32_t crc = UINT32_MAX;
  size_t index;
  for (index = 0; index < count; ++index) {
    uint32_t value = crc ^ bytes[index];
    unsigned bit;
    for (bit = 0; bit < 8U; ++bit) {
      const uint32_t mask = (uint32_t)-(int32_t)(value & UINT32_C(1));
      value = (value >> 1U) ^ (UINT32_C(0x82f63b78) & mask);
    }
    crc = value;
  }
  return ~crc;
}

static void chunk_range(uint64_t total, uint32_t chunks, uint32_t chunk,
                        uint64_t *offset, uint64_t *count) {
  const uint64_t base = total / chunks;
  const uint64_t remainder = total % chunks;
  *count = base + (chunk < remainder ? UINT64_C(1) : UINT64_C(0));
  *offset = (uint64_t)chunk * base +
            (chunk < remainder ? chunk : (uint32_t)remainder);
}

static void initialize_step(sagr_ccl_v1_plan_step_t *step, uint32_t phase,
                            uint32_t action, uint32_t step_index) {
  memset(step, 0, sizeof(*step));
  step->struct_size = (uint32_t)sizeof(*step);
  step->phase = phase;
  step->action = action;
  step->step_index = step_index;
  step->send_rank = SAGR_CCL_V1_NO_RANK;
  step->receive_rank = SAGR_CCL_V1_NO_RANK;
  step->send_chunk = SAGR_CCL_V1_NO_CHUNK;
  step->receive_chunk = SAGR_CCL_V1_NO_CHUNK;
}

static sagr_ccl_v1_status_t abort_state(
    sagr_ccl_v1_group_state_t *state, uint32_t rank,
    sagr_ccl_v1_status_t reason) {
  if (rank != SAGR_CCL_V1_NO_RANK &&
      reason != SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH &&
      rank >= state->identity.world_size) {
    rank = SAGR_CCL_V1_NO_RANK;
  }
  state->phase = SAGR_CCL_V1_GROUP_PHASE_ABORTED;
  state->active_sequence = 0U;
  state->begun_mask = 0U;
  state->completed_mask = 0U;
  memset(&state->active_descriptor, 0, sizeof(state->active_descriptor));
  state->abort_rank = rank;
  state->abort_status = reason;
  return reason;
}

static int allowed_abort_reason(sagr_ccl_v1_status_t reason) {
  return reason == SAGR_CCL_V1_STATUS_TIMED_OUT ||
         reason == SAGR_CCL_V1_STATUS_PEER_LOST ||
         reason == SAGR_CCL_V1_STATUS_CANCELLED ||
         reason == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR ||
         reason == SAGR_CCL_V1_STATUS_CHECKSUM_ERROR ||
         reason == SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH ||
         reason == SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
}

static int recorded_abort_reason(sagr_ccl_v1_status_t reason) {
  return reason == SAGR_CCL_V1_STATUS_INVALID_ARGUMENT ||
         reason == SAGR_CCL_V1_STATUS_VERSION_MISMATCH ||
         reason == SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH ||
         reason == SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH ||
         reason == SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH ||
         reason == SAGR_CCL_V1_STATUS_NOT_SUPPORTED ||
         reason == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR ||
         reason == SAGR_CCL_V1_STATUS_CHECKSUM_ERROR ||
         reason == SAGR_CCL_V1_STATUS_OUT_OF_ORDER ||
         reason == SAGR_CCL_V1_STATUS_TIMED_OUT ||
         reason == SAGR_CCL_V1_STATUS_PEER_LOST ||
         reason == SAGR_CCL_V1_STATUS_CANCELLED;
}

static int descriptor_is_zero(
    const sagr_ccl_v1_descriptor_t *descriptor) {
  return bytes_are_zero((const uint8_t *)descriptor, sizeof(*descriptor));
}

static int abort_record_is_clean(
    const sagr_ccl_v1_group_state_t *state) {
  return state->abort_rank == SAGR_CCL_V1_NO_RANK &&
         state->abort_status == SAGR_CCL_V1_STATUS_SUCCESS;
}

static int abort_record_is_valid(
    const sagr_ccl_v1_group_state_t *state) {
  if (!recorded_abort_reason(state->abort_status)) {
    return 0;
  }
  return state->abort_rank == SAGR_CCL_V1_NO_RANK ||
         state->abort_status == SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH ||
         state->abort_rank < state->identity.world_size;
}

static int active_descriptor_is_valid(
    const sagr_ccl_v1_group_state_t *state) {
  sagr_ccl_v1_descriptor_t descriptor = state->active_descriptor;
  if (descriptor.rank != SAGR_CCL_V1_NO_RANK) {
    return 0;
  }
  descriptor.rank = 0U;
  return sagr_ccl_v1_descriptor_validate(&descriptor) ==
             SAGR_CCL_V1_STATUS_SUCCESS &&
         sagr_ccl_v1_group_identity_equal(&descriptor.group,
                                          &state->identity) &&
         descriptor.sequence == state->active_sequence;
}

static int inactive_collective_is_clean(
    const sagr_ccl_v1_group_state_t *state) {
  return state->active_sequence == 0U && state->begun_mask == 0U &&
         state->completed_mask == 0U &&
         descriptor_is_zero(&state->active_descriptor);
}

static sagr_ccl_v1_status_t validate_state(
    const sagr_ccl_v1_group_state_t *state) {
  uint32_t mask;
  if (state == NULL || state->struct_size != sizeof(*state)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (state->reserved0 != 0U ||
      !bytes_are_zero(state->reserved, sizeof(state->reserved)) ||
      sagr_ccl_v1_group_identity_validate(&state->identity) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      state->next_sequence == 0U || state->next_sequence == UINT64_MAX) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  mask = world_mask(state->identity.world_size);
  if (((state->joined_mask | state->begun_mask | state->completed_mask) &
       ~mask) != 0U ||
      (state->completed_mask & ~state->begun_mask) != 0U) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  switch (state->phase) {
    case SAGR_CCL_V1_GROUP_PHASE_JOINING:
      if (state->joined_mask == mask || !inactive_collective_is_clean(state) ||
          !abort_record_is_clean(state)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_GROUP_PHASE_READY:
      if (state->joined_mask != mask || !inactive_collective_is_clean(state) ||
          !abort_record_is_clean(state)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_GROUP_PHASE_COLLECTING:
      if (state->joined_mask != mask || state->begun_mask == 0U ||
          state->begun_mask == mask || state->completed_mask != 0U ||
          state->active_sequence != state->next_sequence ||
          !active_descriptor_is_valid(state) ||
          !abort_record_is_clean(state)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_GROUP_PHASE_ACTIVE:
      if (state->joined_mask != mask || state->begun_mask != mask ||
          state->completed_mask == mask ||
          state->active_sequence != state->next_sequence ||
          !active_descriptor_is_valid(state) ||
          !abort_record_is_clean(state)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_GROUP_PHASE_ABORTED:
      if (!inactive_collective_is_clean(state) ||
          !abort_record_is_valid(state)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    case SAGR_CCL_V1_GROUP_PHASE_CLOSED:
      if (!inactive_collective_is_clean(state) ||
          (!abort_record_is_clean(state) && !abort_record_is_valid(state)) ||
          (abort_record_is_clean(state) && state->joined_mask != mask)) {
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      break;
    default:
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static sagr_ccl_v1_status_t validate_state_for_mutation(
    sagr_ccl_v1_group_state_t *state) {
  const sagr_ccl_v1_status_t status = validate_state(state);
  if (status == SAGR_CCL_V1_STATUS_SUCCESS || state == NULL ||
      state->struct_size != sizeof(*state)) {
    return status;
  }
  state->reserved0 = 0U;
  memset(state->reserved, 0, sizeof(state->reserved));
  if (sagr_ccl_v1_group_identity_validate(&state->identity) ==
      SAGR_CCL_V1_STATUS_SUCCESS) {
    state->joined_mask &= world_mask(state->identity.world_size);
  } else {
    state->joined_mask = 0U;
  }
  if (state->next_sequence == 0U || state->next_sequence == UINT64_MAX) {
    state->next_sequence = 1U;
  }
  (void)abort_state(state, SAGR_CCL_V1_NO_RANK,
                    SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
}

static int descriptor_collective_equal(
    const sagr_ccl_v1_descriptor_t *left,
    const sagr_ccl_v1_descriptor_t *right) {
  return sagr_ccl_v1_group_identity_equal(&left->group, &right->group) &&
         left->sequence == right->sequence &&
         left->input_count == right->input_count &&
         left->output_count == right->output_count &&
         left->operation == right->operation &&
         left->reduction == right->reduction && left->dtype == right->dtype &&
         left->root_rank == right->root_rank;
}

const char *sagr_ccl_v1_status_string(sagr_ccl_v1_status_t status) {
  switch (status) {
    case SAGR_CCL_V1_STATUS_SUCCESS:
      return "success";
    case SAGR_CCL_V1_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL:
      return "buffer too small";
    case SAGR_CCL_V1_STATUS_VERSION_MISMATCH:
      return "version mismatch";
    case SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH:
      return "topology mismatch";
    case SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH:
      return "identity mismatch";
    case SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH:
      return "sequence mismatch";
    case SAGR_CCL_V1_STATUS_NOT_SUPPORTED:
      return "not supported";
    case SAGR_CCL_V1_STATUS_PROTOCOL_ERROR:
      return "protocol error";
    case SAGR_CCL_V1_STATUS_CHECKSUM_ERROR:
      return "checksum error";
    case SAGR_CCL_V1_STATUS_OUT_OF_ORDER:
      return "out of order";
    case SAGR_CCL_V1_STATUS_BUSY:
      return "busy";
    case SAGR_CCL_V1_STATUS_ABORTED:
      return "aborted";
    case SAGR_CCL_V1_STATUS_TIMED_OUT:
      return "timed out";
    case SAGR_CCL_V1_STATUS_PEER_LOST:
      return "peer lost";
    case SAGR_CCL_V1_STATUS_CANCELLED:
      return "cancelled";
    case SAGR_CCL_V1_STATUS_CLOSED:
      return "closed";
    case SAGR_CCL_V1_STATUS_OUT_OF_RESOURCES:
      return "out of resources";
    default:
      return "unknown status";
  }
}

sagr_ccl_v1_status_t sagr_ccl_v1_group_identity_init(
    sagr_ccl_v1_group_identity_t *identity, uint32_t identity_size) {
  if (identity == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (identity_size < sizeof(*identity)) {
    if (identity_size >= sizeof(identity->struct_size)) {
      identity->struct_size = (uint32_t)sizeof(*identity);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  memset(identity, 0, sizeof(*identity));
  identity->struct_size = (uint32_t)sizeof(*identity);
  identity->protocol_major = SAGR_CCL_V1_PROTOCOL_MAJOR;
  identity->protocol_minor = SAGR_CCL_V1_PROTOCOL_MINOR;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_group_identity_validate(
    const sagr_ccl_v1_group_identity_t *identity) {
  if (identity == NULL || identity->struct_size != sizeof(*identity) ||
      identity->flags != 0U || !bytes_are_zero(identity->reserved,
                                               sizeof(identity->reserved))) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (identity->protocol_major != SAGR_CCL_V1_PROTOCOL_MAJOR ||
      identity->protocol_minor != SAGR_CCL_V1_PROTOCOL_MINOR) {
    return SAGR_CCL_V1_STATUS_VERSION_MISMATCH;
  }
  if (!supported_world_size(identity->world_size)) {
    return SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH;
  }
  if (identity->epoch == 0U || identity->group_generation == 0U ||
      !bytes_are_nonzero(identity->job_uuid, sizeof(identity->job_uuid)) ||
      !bytes_are_nonzero(identity->group_uuid, sizeof(identity->group_uuid)) ||
      !bytes_are_nonzero(identity->model_identity_sha256,
                         sizeof(identity->model_identity_sha256))) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

int sagr_ccl_v1_group_identity_equal(
    const sagr_ccl_v1_group_identity_t *left,
    const sagr_ccl_v1_group_identity_t *right) {
  if (sagr_ccl_v1_group_identity_validate(left) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      sagr_ccl_v1_group_identity_validate(right) !=
          SAGR_CCL_V1_STATUS_SUCCESS) {
    return 0;
  }
  return left->protocol_major == right->protocol_major &&
         left->protocol_minor == right->protocol_minor &&
         left->world_size == right->world_size && left->epoch == right->epoch &&
         left->group_generation == right->group_generation &&
         memcmp(left->job_uuid, right->job_uuid, sizeof(left->job_uuid)) == 0 &&
         memcmp(left->group_uuid, right->group_uuid,
                sizeof(left->group_uuid)) == 0 &&
         memcmp(left->model_identity_sha256, right->model_identity_sha256,
                sizeof(left->model_identity_sha256)) == 0;
}

sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_init(
    sagr_ccl_v1_descriptor_t *descriptor, uint32_t descriptor_size) {
  sagr_ccl_v1_status_t status;
  if (descriptor == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (descriptor_size < sizeof(*descriptor)) {
    if (descriptor_size >= sizeof(descriptor->struct_size)) {
      descriptor->struct_size = (uint32_t)sizeof(*descriptor);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  memset(descriptor, 0, sizeof(*descriptor));
  descriptor->struct_size = (uint32_t)sizeof(*descriptor);
  status = sagr_ccl_v1_group_identity_init(
      &descriptor->group, (uint32_t)sizeof(descriptor->group));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  descriptor->root_rank = SAGR_CCL_V1_NO_RANK;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_validate(
    const sagr_ccl_v1_descriptor_t *descriptor) {
  uint64_t expected;
  sagr_ccl_v1_status_t status;
  if (descriptor == NULL || descriptor->struct_size != sizeof(*descriptor) ||
      descriptor->flags != 0U || descriptor->reserved0 != 0U ||
      !bytes_are_zero(descriptor->reserved, sizeof(descriptor->reserved))) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  status = sagr_ccl_v1_group_identity_validate(&descriptor->group);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (descriptor->rank >= descriptor->group.world_size) {
    return SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH;
  }
  if (descriptor->sequence == 0U || descriptor->sequence == UINT64_MAX) {
    return SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH;
  }
  switch (descriptor->operation) {
    case SAGR_CCL_V1_OPERATION_ALL_REDUCE:
      if (descriptor->reduction != SAGR_CCL_V1_REDUCTION_SUM ||
          !valid_sum_dtype(descriptor->dtype)) {
        return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
      }
      if (descriptor->input_count == 0U ||
          descriptor->input_count != descriptor->output_count ||
          descriptor->root_rank != SAGR_CCL_V1_NO_RANK) {
        return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
      }
      break;
    case SAGR_CCL_V1_OPERATION_ALL_GATHER:
      if (descriptor->reduction != SAGR_CCL_V1_REDUCTION_NONE ||
          !valid_copy_dtype(descriptor->dtype)) {
        return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
      }
      if (!multiply_u64(descriptor->input_count, descriptor->group.world_size,
                        &expected) ||
          descriptor->input_count == 0U || expected != descriptor->output_count ||
          descriptor->root_rank != SAGR_CCL_V1_NO_RANK) {
        return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
      }
      break;
    case SAGR_CCL_V1_OPERATION_REDUCE_SCATTER:
      if (descriptor->reduction != SAGR_CCL_V1_REDUCTION_SUM ||
          !valid_sum_dtype(descriptor->dtype)) {
        return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
      }
      if (!multiply_u64(descriptor->output_count, descriptor->group.world_size,
                        &expected) ||
          descriptor->output_count == 0U || expected != descriptor->input_count ||
          descriptor->root_rank != SAGR_CCL_V1_NO_RANK) {
        return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
      }
      break;
    case SAGR_CCL_V1_OPERATION_BROADCAST:
      if (descriptor->reduction != SAGR_CCL_V1_REDUCTION_NONE ||
          !valid_copy_dtype(descriptor->dtype)) {
        return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
      }
      if (descriptor->input_count == 0U ||
          descriptor->input_count != descriptor->output_count ||
          descriptor->root_rank >= descriptor->group.world_size) {
        return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
      }
      break;
    case SAGR_CCL_V1_OPERATION_BARRIER:
      if (descriptor->reduction != SAGR_CCL_V1_REDUCTION_NONE ||
          descriptor->dtype != SAGR_CCL_V1_DTYPE_NONE ||
          descriptor->input_count != 0U || descriptor->output_count != 0U ||
          descriptor->root_rank != SAGR_CCL_V1_NO_RANK) {
        return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
      }
      break;
    default:
      return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_encode(
    const sagr_ccl_v1_descriptor_t *descriptor, uint8_t *wire,
    uint32_t wire_size) {
  sagr_ccl_v1_status_t status = sagr_ccl_v1_descriptor_validate(descriptor);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (wire == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (wire_size < SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES) {
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  memset(wire, 0, SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES);
  put_u32(wire, SAGR_CCL_V1_WIRE_MAGIC);
  put_u16(wire + 4, descriptor->group.protocol_major);
  put_u16(wire + 6, descriptor->group.protocol_minor);
  put_u32(wire + 8, SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES);
  put_u32(wire + 12, descriptor->flags);
  put_u32(wire + 16, descriptor->operation);
  put_u32(wire + 20, descriptor->reduction);
  put_u32(wire + 24, descriptor->dtype);
  put_u32(wire + 28, descriptor->rank);
  put_u32(wire + 32, descriptor->group.world_size);
  put_u32(wire + 36, descriptor->root_rank);
  put_u64(wire + 40, descriptor->sequence);
  put_u64(wire + 48, descriptor->input_count);
  put_u64(wire + 56, descriptor->output_count);
  put_u64(wire + 64, descriptor->group.epoch);
  put_u64(wire + 72, descriptor->group.group_generation);
  memcpy(wire + 80, descriptor->group.job_uuid,
         sizeof(descriptor->group.job_uuid));
  memcpy(wire + 96, descriptor->group.group_uuid,
         sizeof(descriptor->group.group_uuid));
  memcpy(wire + 112, descriptor->group.model_identity_sha256,
         sizeof(descriptor->group.model_identity_sha256));
  put_u32(wire + SAGR_CCL_V1_WIRE_CRC_OFFSET,
          crc32c(wire, SAGR_CCL_V1_WIRE_CRC_OFFSET));
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_decode(
    const uint8_t *wire, uint32_t wire_size,
    sagr_ccl_v1_descriptor_t *descriptor, uint32_t descriptor_size) {
  sagr_ccl_v1_status_t status;
  uint32_t encoded_crc;
  if (wire == NULL || descriptor == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (wire_size != SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES) {
    return wire_size < SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES
               ? SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL
               : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (descriptor_size < sizeof(*descriptor)) {
    if (descriptor_size >= sizeof(descriptor->struct_size)) {
      descriptor->struct_size = (uint32_t)sizeof(*descriptor);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  if (get_u32(wire) != SAGR_CCL_V1_WIRE_MAGIC ||
      get_u32(wire + 8) != SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES ||
      !bytes_are_zero(wire + SAGR_CCL_V1_WIRE_RESERVED_OFFSET,
                      SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES -
                          SAGR_CCL_V1_WIRE_RESERVED_OFFSET)) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (get_u16(wire + 4) != SAGR_CCL_V1_PROTOCOL_MAJOR ||
      get_u16(wire + 6) != SAGR_CCL_V1_PROTOCOL_MINOR) {
    return SAGR_CCL_V1_STATUS_VERSION_MISMATCH;
  }
  encoded_crc = get_u32(wire + SAGR_CCL_V1_WIRE_CRC_OFFSET);
  if (encoded_crc != crc32c(wire, SAGR_CCL_V1_WIRE_CRC_OFFSET)) {
    return SAGR_CCL_V1_STATUS_CHECKSUM_ERROR;
  }
  status = sagr_ccl_v1_descriptor_init(descriptor, descriptor_size);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  descriptor->flags = get_u32(wire + 12);
  descriptor->operation = get_u32(wire + 16);
  descriptor->reduction = get_u32(wire + 20);
  descriptor->dtype = get_u32(wire + 24);
  descriptor->rank = get_u32(wire + 28);
  descriptor->group.world_size = get_u32(wire + 32);
  descriptor->root_rank = get_u32(wire + 36);
  descriptor->sequence = get_u64(wire + 40);
  descriptor->input_count = get_u64(wire + 48);
  descriptor->output_count = get_u64(wire + 56);
  descriptor->group.epoch = get_u64(wire + 64);
  descriptor->group.group_generation = get_u64(wire + 72);
  memcpy(descriptor->group.job_uuid, wire + 80,
         sizeof(descriptor->group.job_uuid));
  memcpy(descriptor->group.group_uuid, wire + 96,
         sizeof(descriptor->group.group_uuid));
  memcpy(descriptor->group.model_identity_sha256, wire + 112,
         sizeof(descriptor->group.model_identity_sha256));
  return sagr_ccl_v1_descriptor_validate(descriptor);
}

sagr_ccl_v1_status_t sagr_ccl_v1_descriptor_sha256(
    const sagr_ccl_v1_descriptor_t *descriptor,
    uint8_t digest[SAGR_CCL_V1_SHA256_BYTES]) {
  sagr_ccl_v1_descriptor_t canonical;
  uint8_t wire[SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES];
  sagr_ccl_v1_status_t status;
  if (digest == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  status = sagr_ccl_v1_descriptor_validate(descriptor);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    memset(digest, 0, SAGR_CCL_V1_SHA256_BYTES);
    return status;
  }
  canonical = *descriptor;
  canonical.rank = 0U;
  status = sagr_ccl_v1_descriptor_encode(&canonical, wire,
                                         (uint32_t)sizeof(wire));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    memset(digest, 0, SAGR_CCL_V1_SHA256_BYTES);
    return status;
  }
  sagr_sha256(wire, sizeof(wire), digest);
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_plan_required_steps(
    const sagr_ccl_v1_descriptor_t *descriptor, uint32_t *required_steps) {
  uint32_t rounds = 0U;
  uint32_t distance;
  sagr_ccl_v1_status_t status = sagr_ccl_v1_descriptor_validate(descriptor);
  if (required_steps == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  *required_steps = 0U;
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  switch (descriptor->operation) {
    case SAGR_CCL_V1_OPERATION_ALL_REDUCE:
      *required_steps = 2U * (descriptor->group.world_size - 1U);
      break;
    case SAGR_CCL_V1_OPERATION_ALL_GATHER:
    case SAGR_CCL_V1_OPERATION_REDUCE_SCATTER:
    case SAGR_CCL_V1_OPERATION_BROADCAST:
      *required_steps = descriptor->group.world_size - 1U;
      break;
    case SAGR_CCL_V1_OPERATION_BARRIER:
      for (distance = 1U; distance < descriptor->group.world_size;
           distance <<= 1U) {
        ++rounds;
      }
      *required_steps = rounds;
      break;
    default:
      return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static void plan_reduce_scatter(const sagr_ccl_v1_descriptor_t *descriptor,
                                sagr_ccl_v1_plan_step_t *steps,
                                uint32_t start_index) {
  const uint32_t world = descriptor->group.world_size;
  const uint32_t rank = descriptor->rank;
  uint32_t index;
  for (index = 0; index < world - 1U; ++index) {
    const uint32_t send_chunk = (rank + world - index - 1U) % world;
    const uint32_t receive_chunk = (rank + world - index - 2U) % world;
    sagr_ccl_v1_plan_step_t *step = &steps[start_index + index];
    initialize_step(step, SAGR_CCL_V1_PLAN_PHASE_REDUCE_SCATTER,
                    SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE, index);
    step->send_rank = (rank + 1U) % world;
    step->receive_rank = (rank + world - 1U) % world;
    step->send_chunk = send_chunk;
    step->receive_chunk = receive_chunk;
    if (descriptor->operation == SAGR_CCL_V1_OPERATION_ALL_REDUCE) {
      chunk_range(descriptor->input_count, world, send_chunk,
                  &step->send_offset_elements, &step->send_count_elements);
      chunk_range(descriptor->input_count, world, receive_chunk,
                  &step->receive_offset_elements,
                  &step->receive_count_elements);
    } else {
      step->send_offset_elements =
          (uint64_t)send_chunk * descriptor->output_count;
      step->send_count_elements = descriptor->output_count;
      step->receive_offset_elements =
          (uint64_t)receive_chunk * descriptor->output_count;
      step->receive_count_elements = descriptor->output_count;
    }
  }
}

static void plan_all_gather(const sagr_ccl_v1_descriptor_t *descriptor,
                            sagr_ccl_v1_plan_step_t *steps,
                            uint32_t start_index) {
  const uint32_t world = descriptor->group.world_size;
  const uint32_t rank = descriptor->rank;
  const uint64_t chunk_count =
      descriptor->operation == SAGR_CCL_V1_OPERATION_ALL_GATHER
          ? descriptor->input_count
          : 0U;
  uint32_t index;
  for (index = 0; index < world - 1U; ++index) {
    const uint32_t send_chunk = (rank + world - index) % world;
    const uint32_t receive_chunk = (rank + world - index - 1U) % world;
    sagr_ccl_v1_plan_step_t *step = &steps[start_index + index];
    initialize_step(step, SAGR_CCL_V1_PLAN_PHASE_ALL_GATHER,
                    SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE, index);
    step->send_rank = (rank + 1U) % world;
    step->receive_rank = (rank + world - 1U) % world;
    step->send_chunk = send_chunk;
    step->receive_chunk = receive_chunk;
    if (descriptor->operation == SAGR_CCL_V1_OPERATION_ALL_GATHER) {
      step->send_offset_elements = (uint64_t)send_chunk * chunk_count;
      step->send_count_elements = chunk_count;
      step->receive_offset_elements = (uint64_t)receive_chunk * chunk_count;
      step->receive_count_elements = chunk_count;
    } else {
      chunk_range(descriptor->output_count, world, send_chunk,
                  &step->send_offset_elements, &step->send_count_elements);
      chunk_range(descriptor->output_count, world, receive_chunk,
                  &step->receive_offset_elements,
                  &step->receive_count_elements);
    }
  }
}

static void plan_broadcast(const sagr_ccl_v1_descriptor_t *descriptor,
                           sagr_ccl_v1_plan_step_t *steps) {
  const uint32_t world = descriptor->group.world_size;
  uint32_t index;
  for (index = 0; index < world - 1U; ++index) {
    const uint32_t sender = (descriptor->root_rank + index) % world;
    const uint32_t receiver = (sender + 1U) % world;
    sagr_ccl_v1_plan_step_t *step = &steps[index];
    initialize_step(step, SAGR_CCL_V1_PLAN_PHASE_BROADCAST,
                    SAGR_CCL_V1_PLAN_ACTION_IDLE, index);
    step->send_chunk = 0U;
    step->receive_chunk = 0U;
    if (descriptor->rank == sender) {
      step->action = SAGR_CCL_V1_PLAN_ACTION_SEND;
      step->send_rank = receiver;
      step->send_offset_elements = 0U;
      step->send_count_elements = descriptor->input_count;
    } else if (descriptor->rank == receiver) {
      step->action = SAGR_CCL_V1_PLAN_ACTION_RECEIVE;
      step->receive_rank = sender;
      step->receive_offset_elements = 0U;
      step->receive_count_elements = descriptor->output_count;
    }
  }
}

static void plan_barrier(const sagr_ccl_v1_descriptor_t *descriptor,
                         sagr_ccl_v1_plan_step_t *steps,
                         uint32_t step_count) {
  const uint32_t world = descriptor->group.world_size;
  const uint32_t rank = descriptor->rank;
  uint32_t index;
  uint32_t distance = 1U;
  for (index = 0; index < step_count; ++index, distance <<= 1U) {
    sagr_ccl_v1_plan_step_t *step = &steps[index];
    initialize_step(step, SAGR_CCL_V1_PLAN_PHASE_BARRIER,
                    SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE, index);
    step->send_rank = (rank + distance) % world;
    step->receive_rank = (rank + world - (distance % world)) % world;
  }
}

sagr_ccl_v1_status_t sagr_ccl_v1_plan_rank(
    const sagr_ccl_v1_descriptor_t *descriptor,
    sagr_ccl_v1_plan_step_t *steps, uint32_t step_capacity,
    uint32_t *step_count) {
  uint32_t required;
  sagr_ccl_v1_status_t status =
      sagr_ccl_v1_plan_required_steps(descriptor, &required);
  if (step_count == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  *step_count = required;
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (required != 0U && steps == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (step_capacity < required) {
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  memset(steps, 0, (size_t)required * sizeof(*steps));
  switch (descriptor->operation) {
    case SAGR_CCL_V1_OPERATION_ALL_REDUCE:
      plan_reduce_scatter(descriptor, steps, 0U);
      plan_all_gather(descriptor, steps,
                      descriptor->group.world_size - 1U);
      break;
    case SAGR_CCL_V1_OPERATION_ALL_GATHER:
      plan_all_gather(descriptor, steps, 0U);
      break;
    case SAGR_CCL_V1_OPERATION_REDUCE_SCATTER:
      plan_reduce_scatter(descriptor, steps, 0U);
      break;
    case SAGR_CCL_V1_OPERATION_BROADCAST:
      plan_broadcast(descriptor, steps);
      break;
    case SAGR_CCL_V1_OPERATION_BARRIER:
      plan_barrier(descriptor, steps, required);
      break;
    default:
      return SAGR_CCL_V1_STATUS_NOT_SUPPORTED;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_group_state_init(
    sagr_ccl_v1_group_state_t *state, uint32_t state_size,
    const sagr_ccl_v1_group_identity_t *identity) {
  sagr_ccl_v1_status_t status =
      sagr_ccl_v1_group_identity_validate(identity);
  if (state == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (state_size < sizeof(*state)) {
    if (state_size >= sizeof(state->struct_size)) {
      state->struct_size = (uint32_t)sizeof(*state);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  memset(state, 0, sizeof(*state));
  state->struct_size = (uint32_t)sizeof(*state);
  state->phase = SAGR_CCL_V1_GROUP_PHASE_JOINING;
  state->identity = *identity;
  state->next_sequence = 1U;
  state->abort_rank = SAGR_CCL_V1_NO_RANK;
  state->abort_status = SAGR_CCL_V1_STATUS_SUCCESS;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_group_state_join(
    sagr_ccl_v1_group_state_t *state,
    const sagr_ccl_v1_group_identity_t *identity, uint32_t rank) {
  sagr_ccl_v1_status_t status = validate_state_for_mutation(state);
  uint32_t bit;
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED) {
    return SAGR_CCL_V1_STATUS_ABORTED;
  }
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  if (state->phase != SAGR_CCL_V1_GROUP_PHASE_JOINING) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  }
  if (!sagr_ccl_v1_group_identity_equal(&state->identity, identity)) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  }
  if (rank >= state->identity.world_size) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
  }
  bit = UINT32_C(1) << rank;
  if ((state->joined_mask & bit) != 0U) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  }
  state->joined_mask |= bit;
  if (state->joined_mask == world_mask(state->identity.world_size)) {
    state->phase = SAGR_CCL_V1_GROUP_PHASE_READY;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_group_state_begin(
    sagr_ccl_v1_group_state_t *state,
    const sagr_ccl_v1_descriptor_t *descriptor) {
  sagr_ccl_v1_status_t status = validate_state_for_mutation(state);
  uint32_t bit;
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED) {
    return SAGR_CCL_V1_STATUS_ABORTED;
  }
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  if (state->phase != SAGR_CCL_V1_GROUP_PHASE_READY &&
      state->phase != SAGR_CCL_V1_GROUP_PHASE_COLLECTING) {
    return abort_state(state,
                       descriptor == NULL ? SAGR_CCL_V1_NO_RANK
                                          : descriptor->rank,
                       SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  }
  status = sagr_ccl_v1_descriptor_validate(descriptor);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return abort_state(state,
                       descriptor == NULL ? SAGR_CCL_V1_NO_RANK
                                          : descriptor->rank,
                       status);
  }
  if (!sagr_ccl_v1_group_identity_equal(&state->identity,
                                        &descriptor->group)) {
    return abort_state(state, descriptor->rank,
                       SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  }
  if (descriptor->sequence != state->next_sequence) {
    return abort_state(state, descriptor->rank,
                       SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH);
  }
  bit = UINT32_C(1) << descriptor->rank;
  if ((state->joined_mask & bit) == 0U || (state->begun_mask & bit) != 0U) {
    return abort_state(state, descriptor->rank,
                       SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  }
  if (state->begun_mask == 0U) {
    state->active_sequence = descriptor->sequence;
    state->active_descriptor = *descriptor;
    state->active_descriptor.rank = SAGR_CCL_V1_NO_RANK;
    state->phase = SAGR_CCL_V1_GROUP_PHASE_COLLECTING;
  } else if (!descriptor_collective_equal(&state->active_descriptor,
                                          descriptor)) {
    return abort_state(state, descriptor->rank,
                       SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  }
  state->begun_mask |= bit;
  if (state->begun_mask == world_mask(state->identity.world_size)) {
    state->phase = SAGR_CCL_V1_GROUP_PHASE_ACTIVE;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_group_state_complete(
    sagr_ccl_v1_group_state_t *state, uint32_t rank, uint64_t sequence) {
  sagr_ccl_v1_status_t status = validate_state_for_mutation(state);
  uint32_t bit;
  uint32_t complete_mask;
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  complete_mask = world_mask(state->identity.world_size);
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED) {
    return SAGR_CCL_V1_STATUS_ABORTED;
  }
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  if (state->phase != SAGR_CCL_V1_GROUP_PHASE_ACTIVE) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  }
  if (rank >= state->identity.world_size) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
  }
  if (sequence != state->active_sequence) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH);
  }
  bit = UINT32_C(1) << rank;
  if ((state->begun_mask & bit) == 0U ||
      (state->completed_mask & bit) != 0U) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  }
  state->completed_mask |= bit;
  if (state->completed_mask == complete_mask) {
    if (state->next_sequence == UINT64_MAX - UINT64_C(1)) {
      return abort_state(state, rank, SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
    }
    ++state->next_sequence;
    state->active_sequence = 0U;
    state->begun_mask = 0U;
    state->completed_mask = 0U;
    memset(&state->active_descriptor, 0, sizeof(state->active_descriptor));
    state->phase = SAGR_CCL_V1_GROUP_PHASE_READY;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_group_state_abort(
    sagr_ccl_v1_group_state_t *state, uint32_t rank, uint64_t sequence,
    sagr_ccl_v1_status_t reason) {
  sagr_ccl_v1_status_t status = validate_state_for_mutation(state);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED) {
    return SAGR_CCL_V1_STATUS_ABORTED;
  }
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  if (rank >= state->identity.world_size) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
  }
  if (!allowed_abort_reason(reason)) {
    return abort_state(state, rank, SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  }
  if (state->active_sequence != 0U && sequence != state->active_sequence) {
    return abort_state(state, rank,
                       SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH);
  }
  if (state->active_sequence == 0U && sequence != state->next_sequence) {
    return abort_state(state, rank,
                       SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH);
  }
  return abort_state(state, rank, reason);
}

sagr_ccl_v1_status_t sagr_ccl_v1_group_state_close(
    sagr_ccl_v1_group_state_t *state) {
  sagr_ccl_v1_status_t status = validate_state_for_mutation(state);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  if (state->phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED) {
    state->phase = SAGR_CCL_V1_GROUP_PHASE_CLOSED;
    return SAGR_CCL_V1_STATUS_SUCCESS;
  }
  if (state->phase != SAGR_CCL_V1_GROUP_PHASE_READY) {
    return SAGR_CCL_V1_STATUS_BUSY;
  }
  state->phase = SAGR_CCL_V1_GROUP_PHASE_CLOSED;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_v1_group_state_snapshot(
    const sagr_ccl_v1_group_state_t *state,
    sagr_ccl_v1_group_snapshot_t *snapshot, uint32_t snapshot_size) {
  sagr_ccl_v1_status_t status = validate_state(state);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (snapshot == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (snapshot_size < sizeof(*snapshot)) {
    if (snapshot_size >= sizeof(snapshot->struct_size)) {
      snapshot->struct_size = (uint32_t)sizeof(*snapshot);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  memset(snapshot, 0, sizeof(*snapshot));
  snapshot->struct_size = (uint32_t)sizeof(*snapshot);
  snapshot->phase = state->phase;
  snapshot->next_sequence = state->next_sequence;
  snapshot->active_sequence = state->active_sequence;
  snapshot->joined_mask = state->joined_mask;
  snapshot->begun_mask = state->begun_mask;
  snapshot->completed_mask = state->completed_mask;
  snapshot->abort_rank = state->abort_rank;
  snapshot->abort_status = state->abort_status;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}
