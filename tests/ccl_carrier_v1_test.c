/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <self_amdgpu_runtime/ccl_carrier_v1.h>

#include <fcntl.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#define REQUIRE(expression)                                                    \
  do {                                                                         \
    if (!(expression)) {                                                       \
      fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, \
              #expression);                                                    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static sagr_ccl_v1_group_identity_t make_identity(uint32_t world_size) {
  sagr_ccl_v1_group_identity_t identity;
  uint32_t index;
  (void)sagr_ccl_v1_group_identity_init(&identity, (uint32_t)sizeof(identity));
  identity.world_size = world_size;
  identity.epoch = UINT64_C(0x0102030405060708);
  identity.group_generation = UINT64_C(17);
  for (index = 0U; index < SAGR_CCL_V1_UUID_BYTES; ++index) {
    identity.job_uuid[index] = (uint8_t)(index + 1U);
    identity.group_uuid[index] = (uint8_t)(0x30U + index);
  }
  for (index = 0U; index < SAGR_CCL_V1_SHA256_BYTES; ++index) {
    identity.model_identity_sha256[index] = (uint8_t)(0x80U + index);
  }
  return identity;
}

static sagr_ccl_v1_descriptor_t make_descriptor(
    const sagr_ccl_v1_group_identity_t *identity, uint32_t rank) {
  sagr_ccl_v1_descriptor_t descriptor;
  (void)sagr_ccl_v1_descriptor_init(&descriptor,
                                    (uint32_t)sizeof(descriptor));
  descriptor.group = *identity;
  descriptor.sequence = UINT64_C(11);
  descriptor.input_count = UINT64_C(1027);
  descriptor.output_count = UINT64_C(1027);
  descriptor.rank = rank;
  descriptor.operation = SAGR_CCL_V1_OPERATION_ALL_REDUCE;
  descriptor.reduction = SAGR_CCL_V1_REDUCTION_SUM;
  descriptor.dtype = SAGR_CCL_V1_DTYPE_BF16;
  descriptor.root_rank = SAGR_CCL_V1_NO_RANK;
  return descriptor;
}

static sagr_ccl_v1_descriptor_t make_operation_descriptor(
    const sagr_ccl_v1_group_identity_t *identity, uint32_t rank,
    uint32_t operation) {
  sagr_ccl_v1_descriptor_t descriptor = make_descriptor(identity, rank);
  descriptor.operation = operation;
  descriptor.root_rank = SAGR_CCL_V1_NO_RANK;
  switch (operation) {
    case SAGR_CCL_V1_OPERATION_ALL_REDUCE:
      descriptor.input_count = (uint64_t)identity->world_size - 1U;
      descriptor.output_count = descriptor.input_count;
      break;
    case SAGR_CCL_V1_OPERATION_ALL_GATHER:
      descriptor.reduction = SAGR_CCL_V1_REDUCTION_NONE;
      descriptor.dtype = SAGR_CCL_V1_DTYPE_UINT8;
      descriptor.input_count = 3U;
      descriptor.output_count = 3U * identity->world_size;
      break;
    case SAGR_CCL_V1_OPERATION_REDUCE_SCATTER:
      descriptor.output_count = 3U;
      descriptor.input_count = 3U * identity->world_size;
      break;
    case SAGR_CCL_V1_OPERATION_BROADCAST:
      descriptor.reduction = SAGR_CCL_V1_REDUCTION_NONE;
      descriptor.dtype = SAGR_CCL_V1_DTYPE_UINT32;
      descriptor.input_count = 5U;
      descriptor.output_count = 5U;
      descriptor.root_rank = identity->world_size - 1U;
      break;
    case SAGR_CCL_V1_OPERATION_BARRIER:
      descriptor.reduction = SAGR_CCL_V1_REDUCTION_NONE;
      descriptor.dtype = SAGR_CCL_V1_DTYPE_NONE;
      descriptor.input_count = 0U;
      descriptor.output_count = 0U;
      break;
    default:
      break;
  }
  return descriptor;
}

static sagr_ccl_v1_carrier_record_t make_record(
    const sagr_ccl_v1_group_identity_t *identity, uint32_t source,
    uint32_t step_index, uint32_t kind, uint32_t slot_index,
    uint64_t generation, uint32_t payload_crc32c) {
  sagr_ccl_v1_descriptor_t descriptor = make_descriptor(identity, source);
  sagr_ccl_v1_carrier_record_t record;
  memset(&record, 0, sizeof(record));
  (void)sagr_ccl_v1_carrier_record_from_plan(
      &descriptor, step_index, source, kind, slot_index, generation,
      payload_crc32c, &record, (uint32_t)sizeof(record));
  return record;
}

static int test_record_codec_and_descriptor_binding(void) {
  sagr_ccl_v1_group_identity_t identity = make_identity(16U);
  sagr_ccl_v1_descriptor_t rank0 = make_descriptor(&identity, 0U);
  sagr_ccl_v1_descriptor_t rank15 = make_descriptor(&identity, 15U);
  sagr_ccl_v1_carrier_record_t source =
      make_record(&identity, 15U, 2U, SAGR_CCL_V1_CARRIER_MESSAGE_DATA,
                  3U, UINT64_C(7), UINT32_C(0x12345678));
  sagr_ccl_v1_carrier_record_t consumed;
  sagr_ccl_v1_carrier_record_t aborted;
  sagr_ccl_v1_carrier_record_t decoded;
  uint8_t wire[SAGR_CCL_V1_CARRIER_WIRE_BYTES];
  uint8_t second[SAGR_CCL_V1_CARRIER_WIRE_BYTES];
  uint8_t rank0_sha[SAGR_CCL_V1_SHA256_BYTES];
  uint8_t rank15_sha[SAGR_CCL_V1_SHA256_BYTES];
  REQUIRE(sagr_ccl_v1_descriptor_sha256(&rank0, rank0_sha) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_descriptor_sha256(&rank15, rank15_sha) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(memcmp(rank0_sha, rank15_sha, sizeof(rank0_sha)) == 0);
  REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
              &source, &rank0, 2U, 15U) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
              &source, &rank15, 2U, 0U) ==
          SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
  {
    sagr_ccl_v1_descriptor_t unrelated = make_descriptor(&identity, 7U);
    REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
                &source, &unrelated, 2U, 15U) ==
            SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
  }
  REQUIRE(sagr_ccl_v1_carrier_record_from_plan(
              &rank0, 2U, 15U, SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED, 3U,
              UINT64_C(7), 0U, &consumed,
              (uint32_t)sizeof(consumed)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
              &consumed, &rank15, 2U, 0U) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_abort_record(
              &rank15, 15U, 4U, SAGR_CCL_V1_STATUS_PEER_LOST, &aborted,
              (uint32_t)sizeof(aborted)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
              &aborted, &rank0, SAGR_CCL_V1_NO_CHUNK, 15U) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  ++rank15.input_count;
  ++rank15.output_count;
  REQUIRE(sagr_ccl_v1_descriptor_sha256(&rank15, rank15_sha) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(memcmp(rank0_sha, rank15_sha, sizeof(rank0_sha)) != 0);
  REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
              &source, &rank15, 2U, 0U) ==
          SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH);
  REQUIRE(sagr_ccl_v1_carrier_record_validate(&source) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_record_encode(&source, wire, sizeof(wire)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_record_decode(
              wire, sizeof(wire), &decoded, (uint32_t)sizeof(decoded)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_record_encode(&decoded, second,
                                             sizeof(second)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(decoded.group.struct_size == sizeof(decoded.group) &&
          decoded.group.protocol_major == SAGR_CCL_V1_PROTOCOL_MAJOR &&
          decoded.group.protocol_minor == SAGR_CCL_V1_PROTOCOL_MINOR);
  REQUIRE(memcmp(wire, second, sizeof(wire)) == 0);

  second[160] ^= UINT8_C(1);
  REQUIRE(sagr_ccl_v1_carrier_record_decode(
              second, sizeof(second), &decoded,
              (uint32_t)sizeof(decoded)) ==
          SAGR_CCL_V1_STATUS_CHECKSUM_ERROR);
  memcpy(second, wire, sizeof(second));
  second[200] = UINT8_C(1);
  REQUIRE(sagr_ccl_v1_carrier_record_decode(
              second, sizeof(second), &decoded,
              (uint32_t)sizeof(decoded)) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  memcpy(second, wire, sizeof(second));
  second[5] = UINT8_C(2);
  REQUIRE(sagr_ccl_v1_carrier_record_decode(
              second, sizeof(second), &decoded,
              (uint32_t)sizeof(decoded)) ==
          SAGR_CCL_V1_STATUS_VERSION_MISMATCH);

  source.destination_rank = source.source_rank;
  REQUIRE(sagr_ccl_v1_carrier_record_validate(&source) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  source.destination_rank = 0U;
  source.slot_generation = 0U;
  REQUIRE(sagr_ccl_v1_carrier_record_validate(&source) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  source.slot_generation = 7U;
  source.payload_bytes = 0U;
  source.payload_crc32c = 1U;
  REQUIRE(sagr_ccl_v1_carrier_record_validate(&source) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  source.payload_crc32c = 0U;
  REQUIRE(sagr_ccl_v1_carrier_record_validate(&source) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  return 0;
}

static int test_all_world_plan_record_symmetry(void) {
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    sagr_ccl_v1_group_identity_t identity = make_identity(world);
    uint32_t operation;
    for (operation = SAGR_CCL_V1_OPERATION_ALL_REDUCE;
         operation <= SAGR_CCL_V1_OPERATION_BARRIER; ++operation) {
      uint32_t source;
      for (source = 0U; source < world; ++source) {
        sagr_ccl_v1_descriptor_t source_descriptor =
            make_operation_descriptor(&identity, source, operation);
        sagr_ccl_v1_plan_step_t steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
        uint32_t step_count = 0U;
        uint32_t step;
        REQUIRE(sagr_ccl_v1_plan_rank(
                    &source_descriptor, steps, SAGR_CCL_V1_MAX_PLAN_STEPS,
                    &step_count) == SAGR_CCL_V1_STATUS_SUCCESS);
        for (step = 0U; step < step_count; ++step) {
          sagr_ccl_v1_carrier_record_t data;
          sagr_ccl_v1_carrier_record_t consumed;
          sagr_ccl_v1_descriptor_t destination_descriptor;
          if (steps[step].action != SAGR_CCL_V1_PLAN_ACTION_SEND &&
              steps[step].action != SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE) {
            continue;
          }
          destination_descriptor = make_operation_descriptor(
              &identity, steps[step].send_rank, operation);
          REQUIRE(sagr_ccl_v1_carrier_record_from_plan(
                      &source_descriptor, step, source,
                      SAGR_CCL_V1_CARRIER_MESSAGE_DATA, 15U,
                      UINT64_C(19), 0U, &data,
                      (uint32_t)sizeof(data)) ==
                  SAGR_CCL_V1_STATUS_SUCCESS);
          REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
                      &data, &destination_descriptor, step, source) ==
                  SAGR_CCL_V1_STATUS_SUCCESS);
          REQUIRE(sagr_ccl_v1_carrier_record_from_plan(
                      &destination_descriptor, step, source,
                      SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED, 15U,
                      UINT64_C(19), 0U, &consumed,
                      (uint32_t)sizeof(consumed)) ==
                  SAGR_CCL_V1_STATUS_SUCCESS);
          REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
                      &consumed, &source_descriptor, step,
                      steps[step].send_rank) == SAGR_CCL_V1_STATUS_SUCCESS);
          data.source_rank = data.destination_rank;
          REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
                      &data, &destination_descriptor, step, source) !=
                  SAGR_CCL_V1_STATUS_SUCCESS);
        }
      }
    }
  }
  return 0;
}

static int test_plan_record_mutation_matrix(void) {
  sagr_ccl_v1_group_identity_t identity = make_identity(16U);
  sagr_ccl_v1_descriptor_t source =
      make_operation_descriptor(&identity, 15U,
                                SAGR_CCL_V1_OPERATION_ALL_REDUCE);
  sagr_ccl_v1_plan_step_t steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
  sagr_ccl_v1_descriptor_t destination;
  sagr_ccl_v1_carrier_record_t original;
  sagr_ccl_v1_carrier_record_t changed;
  uint32_t step_count = 0U;
  REQUIRE(sagr_ccl_v1_plan_rank(&source, steps, SAGR_CCL_V1_MAX_PLAN_STEPS,
                                &step_count) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(step_count == 30U);
  destination = make_operation_descriptor(
      &identity, steps[2].send_rank, SAGR_CCL_V1_OPERATION_ALL_REDUCE);
  REQUIRE(sagr_ccl_v1_carrier_record_from_plan(
              &source, 2U, 15U, SAGR_CCL_V1_CARRIER_MESSAGE_DATA, 4U, 9U,
              0U, &original, (uint32_t)sizeof(original)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
#define REQUIRE_CHANGED_REJECTED(statement)                                   \
  do {                                                                         \
    changed = original;                                                        \
    statement;                                                                 \
    REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(                    \
                &changed, &destination, 2U, 15U) !=                            \
            SAGR_CCL_V1_STATUS_SUCCESS);                                       \
  } while (0)
  REQUIRE_CHANGED_REJECTED(changed.group.epoch++);
  REQUIRE_CHANGED_REJECTED(changed.group.group_generation++);
  REQUIRE_CHANGED_REJECTED(changed.sequence++);
  REQUIRE_CHANGED_REJECTED(changed.descriptor_sha256[0] ^= UINT8_C(1));
  REQUIRE_CHANGED_REJECTED(changed.phase = SAGR_CCL_V1_PLAN_PHASE_ALL_GATHER);
  REQUIRE_CHANGED_REJECTED(changed.step_index++);
  REQUIRE_CHANGED_REJECTED(changed.chunk_index++);
  REQUIRE_CHANGED_REJECTED(changed.source_rank = 14U);
  REQUIRE_CHANGED_REJECTED(changed.destination_rank = 2U);
  REQUIRE_CHANGED_REJECTED(changed.payload_bytes += 2U);
  REQUIRE_CHANGED_REJECTED(changed.slot_generation = 0U);
  REQUIRE_CHANGED_REJECTED(changed.slot_index = 16U);
#undef REQUIRE_CHANGED_REJECTED
  REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
              &original, &destination, 3U, 15U) ==
          SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
  REQUIRE(sagr_ccl_v1_carrier_record_validate_descriptor(
              &original, &destination, 2U, 14U) ==
          SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
  return 0;
}

static int test_credit_depths(void) {
  static const uint32_t depths[] = {1U, 4U, 16U};
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    sagr_ccl_v1_group_identity_t identity = make_identity(world);
    size_t depth_index;
    for (depth_index = 0U; depth_index < sizeof(depths) / sizeof(depths[0]);
         ++depth_index) {
      sagr_ccl_v1_credit_state_t state;
      uint32_t peer;
      REQUIRE(sagr_ccl_v1_credit_state_init(
                  &state, (uint32_t)sizeof(state), &identity, 0U,
                  depths[depth_index]) == SAGR_CCL_V1_STATUS_SUCCESS);
      for (peer = 1U; peer < world; ++peer) {
        uint32_t slots[SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
        uint64_t generations[SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
        uint32_t index;
        for (index = 0U; index < depths[depth_index]; ++index) {
          REQUIRE(sagr_ccl_v1_credit_acquire(
                      &state, peer, &slots[index], &generations[index]) ==
                  SAGR_CCL_V1_STATUS_SUCCESS);
        }
        for (index = depths[depth_index]; index > 0U; --index) {
          REQUIRE(sagr_ccl_v1_credit_release(
                      &state, peer, slots[index - 1U],
                      generations[index - 1U]) ==
                  SAGR_CCL_V1_STATUS_SUCCESS);
        }
      }
      REQUIRE(sagr_ccl_v1_credit_state_close(&state) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
    }
  }
  return 0;
}

static int test_credit_matrix(void) {
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    uint32_t self;
    sagr_ccl_v1_group_identity_t identity = make_identity(world);
    for (self = 0U; self < world; ++self) {
      sagr_ccl_v1_credit_state_t state;
      uint32_t peer;
      REQUIRE(sagr_ccl_v1_credit_state_init(
                  &state, (uint32_t)sizeof(state), &identity, self,
                  SAGR_CCL_V1_MAX_CREDITS_PER_PEER) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
      for (peer = 0U; peer < world; ++peer) {
        uint32_t slots[SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
        uint64_t generations[SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
        uint32_t index;
        uint32_t ignored_slot = 0U;
        uint64_t ignored_generation = 0U;
        if (peer == self) {
          REQUIRE(sagr_ccl_v1_credit_acquire(
                      &state, peer, &ignored_slot, &ignored_generation) ==
                  SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
          continue;
        }
        for (index = 0U; index < SAGR_CCL_V1_MAX_CREDITS_PER_PEER;
             ++index) {
          REQUIRE(sagr_ccl_v1_credit_acquire(
                      &state, peer, &slots[index], &generations[index]) ==
                  SAGR_CCL_V1_STATUS_SUCCESS);
          REQUIRE(slots[index] == index && generations[index] != 0U);
        }
        REQUIRE(sagr_ccl_v1_credit_acquire(
                    &state, peer, &ignored_slot, &ignored_generation) ==
                SAGR_CCL_V1_STATUS_BUSY);
        REQUIRE(sagr_ccl_v1_credit_release(
                    &state, peer, slots[0], generations[0] + UINT64_C(1)) ==
                SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH);
        for (index = 0U; index < SAGR_CCL_V1_MAX_CREDITS_PER_PEER;
             ++index) {
          REQUIRE(sagr_ccl_v1_credit_release(
                      &state, peer, slots[index], generations[index]) ==
                  SAGR_CCL_V1_STATUS_SUCCESS);
        }
        REQUIRE(sagr_ccl_v1_credit_release(
                    &state, peer, slots[0], generations[0]) ==
                SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
        REQUIRE(sagr_ccl_v1_credit_acquire(
                    &state, peer, &ignored_slot, &ignored_generation) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
        REQUIRE(ignored_slot == 0U &&
                ignored_generation > generations[0]);
        REQUIRE(sagr_ccl_v1_credit_state_close(&state) ==
                SAGR_CCL_V1_STATUS_BUSY);
        REQUIRE(sagr_ccl_v1_credit_release(
                    &state, peer, ignored_slot, ignored_generation) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
      }
      REQUIRE(sagr_ccl_v1_credit_state_close(&state) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(sagr_ccl_v1_credit_state_close(&state) ==
              SAGR_CCL_V1_STATUS_CLOSED);
      state.group.epoch = 1U;
      REQUIRE(sagr_ccl_v1_credit_state_close(&state) ==
              SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
    }
  }
  return 0;
}

static int test_world16_simultaneous_credit_capacity(void) {
  sagr_ccl_v1_group_identity_t identity = make_identity(16U);
  sagr_ccl_v1_credit_state_t state;
  uint32_t slots[16U][SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
  uint64_t generations[16U][SAGR_CCL_V1_MAX_CREDITS_PER_PEER];
  uint32_t peer;
  uint32_t index;
  uint32_t ignored_slot = 0U;
  uint64_t ignored_generation = 0U;
  REQUIRE(sagr_ccl_v1_credit_state_init(
              &state, (uint32_t)sizeof(state), &identity, 0U,
              SAGR_CCL_V1_MAX_CREDITS_PER_PEER) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  for (peer = 1U; peer < identity.world_size; ++peer) {
    for (index = 0U; index < SAGR_CCL_V1_MAX_CREDITS_PER_PEER; ++index) {
      REQUIRE(sagr_ccl_v1_credit_acquire(
                  &state, peer, &slots[peer][index],
                  &generations[peer][index]) == SAGR_CCL_V1_STATUS_SUCCESS);
    }
  }
  for (peer = 1U; peer < identity.world_size; ++peer) {
    REQUIRE(sagr_ccl_v1_credit_acquire(
                &state, peer, &ignored_slot, &ignored_generation) ==
            SAGR_CCL_V1_STATUS_BUSY);
  }
  for (peer = identity.world_size - 1U; peer > 0U; --peer) {
    for (index = SAGR_CCL_V1_MAX_CREDITS_PER_PEER; index > 0U; --index) {
      REQUIRE(sagr_ccl_v1_credit_release(
                  &state, peer, slots[peer][index - 1U],
                  generations[peer][index - 1U]) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
    }
  }
  REQUIRE(sagr_ccl_v1_credit_state_close(&state) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  return 0;
}

static int test_slot_lifecycle_and_faults(void) {
  sagr_ccl_v1_group_identity_t identity = make_identity(3U);
  sagr_ccl_v1_carrier_record_t record = make_record(
      &identity, 0U, 0U, SAGR_CCL_V1_CARRIER_MESSAGE_DATA, 0U, 1U,
      UINT32_C(17));
  sagr_ccl_v1_carrier_record_t stale = record;
  sagr_ccl_v1_carrier_slot_state_t slot;
  REQUIRE(sagr_ccl_v1_carrier_slot_init(&slot, (uint32_t)sizeof(slot)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_slot_consume(&slot, &record) ==
          SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  REQUIRE(sagr_ccl_v1_carrier_slot_publish(&slot, &record) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_slot_publish(&slot, &record) ==
          SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  stale.slot_generation += UINT64_C(1);
  REQUIRE(sagr_ccl_v1_carrier_slot_consume(&slot, &stale) ==
          SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH);
  REQUIRE(sagr_ccl_v1_carrier_slot_consume(&slot, &record) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_slot_consume(&slot, &record) ==
          SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  REQUIRE(sagr_ccl_v1_carrier_slot_release(&slot, &stale) ==
          SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH);
  REQUIRE(sagr_ccl_v1_carrier_slot_release(&slot, &record) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_slot_release(&slot, &record) ==
          SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  REQUIRE(sagr_ccl_v1_carrier_slot_publish(&slot, &record) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_slot_abort(
              &slot, SAGR_CCL_V1_STATUS_PEER_LOST) ==
          SAGR_CCL_V1_STATUS_PEER_LOST);
  REQUIRE(sagr_ccl_v1_carrier_slot_abort(
              &slot, SAGR_CCL_V1_STATUS_TIMED_OUT) ==
          SAGR_CCL_V1_STATUS_PEER_LOST);
  REQUIRE(sagr_ccl_v1_carrier_slot_publish(&slot, &record) ==
          SAGR_CCL_V1_STATUS_ABORTED);
  REQUIRE(sagr_ccl_v1_carrier_slot_close(&slot) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_slot_close(&slot) ==
          SAGR_CCL_V1_STATUS_CLOSED);

  REQUIRE(sagr_ccl_v1_carrier_slot_init(&slot, (uint32_t)sizeof(slot)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  slot.reserved[0] = UINT8_C(1);
  REQUIRE(sagr_ccl_v1_carrier_slot_publish(&slot, &record) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  return 0;
}

static int send_raw(int socket_descriptor, const uint8_t *wire,
                    const int *descriptors, size_t descriptor_count) {
  struct iovec vector;
  struct msghdr message;
  struct cmsghdr *control_message;
  unsigned char control[CMSG_SPACE(sizeof(int) * 2U)];
  memset(&message, 0, sizeof(message));
  memset(control, 0, sizeof(control));
  vector.iov_base = (void *)wire;
  vector.iov_len = SAGR_CCL_V1_CARRIER_WIRE_BYTES;
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  if (descriptor_count != 0U) {
    message.msg_control = control;
    message.msg_controllen = CMSG_SPACE(sizeof(int) * descriptor_count);
    control_message = CMSG_FIRSTHDR(&message);
    if (control_message == NULL) {
      return -1;
    }
    control_message->cmsg_level = SOL_SOCKET;
    control_message->cmsg_type = SCM_RIGHTS;
    control_message->cmsg_len = CMSG_LEN(sizeof(int) * descriptor_count);
    memcpy(CMSG_DATA(control_message), descriptors,
           sizeof(int) * descriptor_count);
  }
  return sendmsg(socket_descriptor, &message, MSG_NOSIGNAL) ==
                 (ssize_t)SAGR_CCL_V1_CARRIER_WIRE_BYTES
             ? 0
             : -1;
}

static int send_empty_with_descriptors(int socket_descriptor,
                                       const int *descriptors,
                                       size_t descriptor_count) {
  struct iovec vector;
  struct msghdr message;
  struct cmsghdr *control_message;
  unsigned char control[CMSG_SPACE(sizeof(int) * 8U)];
  uint8_t empty = 0U;
  if (descriptor_count == 0U || descriptor_count > 8U) {
    return -1;
  }
  memset(&message, 0, sizeof(message));
  memset(control, 0, sizeof(control));
  vector.iov_base = &empty;
  vector.iov_len = 0U;
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  message.msg_control = control;
  message.msg_controllen = CMSG_SPACE(sizeof(int) * descriptor_count);
  control_message = CMSG_FIRSTHDR(&message);
  if (control_message == NULL) {
    return -1;
  }
  control_message->cmsg_level = SOL_SOCKET;
  control_message->cmsg_type = SCM_RIGHTS;
  control_message->cmsg_len = CMSG_LEN(sizeof(int) * descriptor_count);
  memcpy(CMSG_DATA(control_message), descriptors,
         sizeof(int) * descriptor_count);
  return sendmsg(socket_descriptor, &message, MSG_NOSIGNAL) == 0 ? 0 : -1;
}

static int count_open_descriptors(void) {
  DIR *directory = opendir("/proc/self/fd");
  struct dirent *entry;
  int count = 0;
  if (directory == NULL) {
    return -1;
  }
  while ((entry = readdir(directory)) != NULL) {
    if (strcmp(entry->d_name, ".") != 0 &&
        strcmp(entry->d_name, "..") != 0) {
      ++count;
    }
  }
  (void)closedir(directory);
  return count;
}

static int test_payload_and_seqpacket_transport(void) {
  static const uint8_t payload[] = {
      0x00U, 0x01U, 0x7fU, 0x80U, 0xfeU, 0xffU, 0x22U, 0x44U};
  sagr_ccl_v1_group_identity_t identity = make_identity(5U);
  sagr_ccl_v1_descriptor_t transport_descriptor = make_descriptor(&identity, 4U);
  sagr_ccl_v1_carrier_record_t record;
  sagr_ccl_v1_carrier_record_t received_record;
  uint8_t copy[sizeof(payload)];
  uint8_t wire[SAGR_CCL_V1_CARRIER_WIRE_BYTES];
  int sockets[2] = {-1, -1};
  int payload_fd = -1;
  int received_fd = -1;
  int unsealed_fd = -1;
  int descriptor_count_before;
  uint32_t payload_crc = 0U;
  transport_descriptor.input_count = 20U;
  transport_descriptor.output_count = 20U;
  REQUIRE(sagr_ccl_v1_carrier_payload_create(
              payload, sizeof(payload), &payload_fd, &payload_crc) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(payload_fd >= 0 && payload_crc != 0U);
  REQUIRE(sagr_ccl_v1_carrier_payload_validate(
              payload_fd, sizeof(payload), payload_crc) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_payload_validate(
              payload_fd, sizeof(payload), payload_crc ^ UINT32_C(1)) ==
          SAGR_CCL_V1_STATUS_CHECKSUM_ERROR);
  REQUIRE(sagr_ccl_v1_carrier_payload_copy(
              payload_fd, sizeof(payload), payload_crc, copy,
              sizeof(copy)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(memcmp(copy, payload, sizeof(payload)) == 0);
  REQUIRE(sagr_ccl_v1_carrier_payload_copy(
              payload_fd, sizeof(payload), payload_crc, copy,
              sizeof(copy) - 1U) == SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL);

  unsealed_fd = memfd_create("unsealed", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  REQUIRE(unsealed_fd >= 0 && fchmod(unsealed_fd, (mode_t)0600) == 0 &&
          ftruncate(unsealed_fd, (off_t)sizeof(payload)) == 0);
  REQUIRE(sagr_ccl_v1_carrier_payload_validate(
              unsealed_fd, sizeof(payload), payload_crc) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);

  REQUIRE(socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     sockets) == 0);
  descriptor_count_before = count_open_descriptors();
  REQUIRE(descriptor_count_before >= 0);
  {
    int zero_descriptors[1] = {payload_fd};
    received_fd = 12345;
    memset(&received_record, 0x5a, sizeof(received_record));
    REQUIRE(send_empty_with_descriptors(sockets[0], zero_descriptors, 1U) ==
            0);
    REQUIRE(sagr_ccl_v1_carrier_receive(
                sockets[1], &received_record,
                (uint32_t)sizeof(received_record), &received_fd) ==
            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
    REQUIRE(received_fd == 12345);
    REQUIRE(count_open_descriptors() == descriptor_count_before);
  }
  {
    int excess_descriptors[6] = {payload_fd, payload_fd, payload_fd,
                                 payload_fd, payload_fd, payload_fd};
    received_fd = 23456;
    memset(&received_record, 0xa5, sizeof(received_record));
    REQUIRE(send_empty_with_descriptors(sockets[0], excess_descriptors, 6U) ==
            0);
    REQUIRE(sagr_ccl_v1_carrier_receive(
                sockets[1], &received_record,
                (uint32_t)sizeof(received_record), &received_fd) ==
            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
    REQUIRE(received_fd == 23456);
    REQUIRE(count_open_descriptors() == descriptor_count_before);
  }
  REQUIRE(sagr_ccl_v1_carrier_record_from_plan(
              &transport_descriptor, 0U, 4U,
              SAGR_CCL_V1_CARRIER_MESSAGE_DATA, 0U, 9U, payload_crc,
              &record, (uint32_t)sizeof(record)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(record.payload_bytes == sizeof(payload));
  REQUIRE(sagr_ccl_v1_carrier_send(sockets[0], &record, payload_fd) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_receive(
              sockets[1], &received_record,
              (uint32_t)sizeof(received_record) - 1U, &received_fd) ==
          SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL);
  REQUIRE(sagr_ccl_v1_carrier_receive(
              sockets[1], &received_record,
              (uint32_t)sizeof(received_record), &received_fd) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(received_fd >= 0);
  memset(copy, 0, sizeof(copy));
  REQUIRE(sagr_ccl_v1_carrier_payload_copy(
              received_fd, received_record.payload_bytes,
              received_record.payload_crc32c, copy,
              sizeof(copy)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(memcmp(copy, payload, sizeof(payload)) == 0);
  REQUIRE(close(received_fd) == 0);
  received_fd = -1;

  REQUIRE(sagr_ccl_v1_carrier_record_encode(&record, wire, sizeof(wire)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(send_raw(sockets[0], wire, NULL, 0U) == 0);
  REQUIRE(sagr_ccl_v1_carrier_receive(
              sockets[1], &received_record,
              (uint32_t)sizeof(received_record), &received_fd) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  {
    int descriptors[2] = {payload_fd, payload_fd};
    REQUIRE(send_raw(sockets[0], wire, descriptors, 2U) == 0);
    REQUIRE(sagr_ccl_v1_carrier_receive(
                sockets[1], &received_record,
                (uint32_t)sizeof(received_record), &received_fd) ==
            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  }
  wire[160] ^= UINT8_C(1);
  REQUIRE(send_raw(sockets[0], wire, &payload_fd, 1U) == 0);
  REQUIRE(sagr_ccl_v1_carrier_receive(
              sockets[1], &received_record,
              (uint32_t)sizeof(received_record), &received_fd) ==
          SAGR_CCL_V1_STATUS_CHECKSUM_ERROR);

  transport_descriptor.rank = record.destination_rank;
  REQUIRE(sagr_ccl_v1_carrier_record_from_plan(
              &transport_descriptor, 0U, 4U,
              SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED, 0U, 9U, 0U, &record,
              (uint32_t)sizeof(record)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_send(sockets[0], &record, -1) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_receive(
              sockets[1], &received_record,
              (uint32_t)sizeof(received_record), &received_fd) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(received_fd == -1 && received_record.payload_bytes == 0U);

  REQUIRE(close(sockets[0]) == 0);
  sockets[0] = -1;
  REQUIRE(sagr_ccl_v1_carrier_receive(
              sockets[1], &received_record,
              (uint32_t)sizeof(received_record), &received_fd) ==
          SAGR_CCL_V1_STATUS_PEER_LOST);
  REQUIRE(close(sockets[1]) == 0);
  REQUIRE(close(payload_fd) == 0);
  REQUIRE(close(unsealed_fd) == 0);
  return 0;
}

static int test_session_lifecycle_and_abort(void) {
  static const uint8_t payload[4] = {0x11U, 0x22U, 0x33U, 0x44U};
  sagr_ccl_v1_group_identity_t identity = make_identity(3U);
  sagr_ccl_v1_descriptor_t rank0 = make_descriptor(&identity, 0U);
  sagr_ccl_v1_descriptor_t rank1 = make_descriptor(&identity, 1U);
  sagr_ccl_v1_carrier_session_t sender = NULL;
  sagr_ccl_v1_carrier_session_t receiver = NULL;
  sagr_ccl_v1_carrier_record_t data;
  sagr_ccl_v1_carrier_record_t received;
  sagr_ccl_v1_carrier_record_t consumed;
  sagr_ccl_v1_carrier_record_t aborted;
  sagr_ccl_v1_carrier_session_info_t info;
  uint8_t output[sizeof(payload)] = {0U};
  int sockets[2] = {-1, -1};
  int descriptor_count_before;
  rank0.input_count = sizeof(payload) / sizeof(uint16_t) * identity.world_size;
  rank0.output_count = rank0.input_count;
  rank1.input_count = rank0.input_count;
  rank1.output_count = rank0.output_count;
  REQUIRE(socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     sockets) == 0);
  descriptor_count_before = count_open_descriptors();
  REQUIRE(descriptor_count_before >= 0);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 0U, 4U, &sender) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 1U, 4U, &receiver) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &rank1, 0U, 0U, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_BUSY);
  REQUIRE(sagr_ccl_v1_carrier_session_prepare_data(
              sender, &rank0, 0U, payload, sizeof(payload), &data,
              (uint32_t)sizeof(data)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(count_open_descriptors() == descriptor_count_before + 1);
  REQUIRE(sagr_ccl_v1_carrier_session_send_data(sender, sockets[0], &data) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(count_open_descriptors() == descriptor_count_before);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &rank1, 0U, 0U, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(count_open_descriptors() == descriptor_count_before + 1);
  REQUIRE(sagr_ccl_v1_carrier_session_consume(
              receiver, &rank1, 0U, &received, output, sizeof(output),
              &consumed, (uint32_t)sizeof(consumed)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(memcmp(output, payload, sizeof(payload)) == 0);
  REQUIRE(count_open_descriptors() == descriptor_count_before);
  REQUIRE(sagr_ccl_v1_carrier_session_send_consumed(
              receiver, sockets[1], &consumed) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              sender, sockets[0], &rank0, 0U, 1U, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(received.kind == SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED);
  REQUIRE(sagr_ccl_v1_carrier_session_info(
              sender, &info, (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(info.phase == SAGR_CCL_V1_CARRIER_SESSION_RUNNING &&
          info.sender_inflight == 0U && info.receiver_ready == 0U &&
          info.receiver_consumed == 0U);

  ++rank0.sequence;
  ++rank1.sequence;
  REQUIRE(sagr_ccl_v1_carrier_session_prepare_data(
              sender, &rank0, 0U, payload, sizeof(payload), &data,
              (uint32_t)sizeof(data)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(count_open_descriptors() == descriptor_count_before + 1);
  REQUIRE(sagr_ccl_v1_carrier_session_abort(
              sender, &rank0, 1U, SAGR_CCL_V1_STATUS_TIMED_OUT, &aborted,
              (uint32_t)sizeof(aborted)) == SAGR_CCL_V1_STATUS_TIMED_OUT);
  REQUIRE(count_open_descriptors() == descriptor_count_before);
  REQUIRE(sagr_ccl_v1_carrier_session_info(
              sender, &info, (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(info.phase == SAGR_CCL_V1_CARRIER_SESSION_ABORTED &&
          info.first_error == SAGR_CCL_V1_STATUS_TIMED_OUT &&
          info.failed_rank == 1U && info.sender_inflight == 0U);
  {
    sagr_ccl_v1_carrier_record_t stored_abort;
    REQUIRE(sagr_ccl_v1_carrier_session_get_abort(
                sender, &stored_abort, (uint32_t)sizeof(stored_abort)) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(memcmp(&stored_abort, &aborted, sizeof(stored_abort)) == 0);
  }
  {
    sagr_ccl_v1_carrier_record_t wrong_abort;
    sagr_ccl_v1_descriptor_t wrong_descriptor = rank0;
    ++wrong_descriptor.sequence;
    REQUIRE(sagr_ccl_v1_carrier_abort_record(
                &wrong_descriptor, 0U, 1U, SAGR_CCL_V1_STATUS_TIMED_OUT,
                &wrong_abort, (uint32_t)sizeof(wrong_abort)) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(sagr_ccl_v1_carrier_session_send_abort(
                sender, sockets[0], &wrong_abort) ==
            SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  }
  REQUIRE(sagr_ccl_v1_carrier_send(sockets[1], &consumed, -1) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              sender, sockets[0], &rank0, 0U, 1U, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_ABORTED);
  REQUIRE(sagr_ccl_v1_carrier_session_info(
              sender, &info, (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(info.phase == SAGR_CCL_V1_CARRIER_SESSION_ABORTED &&
          info.sender_inflight == 0U &&
          info.first_error == SAGR_CCL_V1_STATUS_TIMED_OUT);
  REQUIRE(sagr_ccl_v1_carrier_session_send_abort(
              sender, sockets[0], &aborted) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &rank1, SAGR_CCL_V1_NO_CHUNK, 0U,
              &received, (uint32_t)sizeof(received)) ==
          SAGR_CCL_V1_STATUS_TIMED_OUT);
  REQUIRE(sagr_ccl_v1_carrier_session_info(
              receiver, &info, (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(info.phase == SAGR_CCL_V1_CARRIER_SESSION_ABORTED &&
          info.first_error == SAGR_CCL_V1_STATUS_TIMED_OUT &&
          info.failed_rank == 1U);
  {
    int relay_sockets[2] = {-1, -1};
    sagr_ccl_v1_descriptor_t rank2 = rank0;
    sagr_ccl_v1_carrier_session_t relay_receiver = NULL;
    sagr_ccl_v1_carrier_record_t stored_abort;
    sagr_ccl_v1_carrier_record_t relayed_abort;
    rank2.rank = 2U;
    REQUIRE(sagr_ccl_v1_carrier_session_get_abort(
                receiver, &stored_abort, (uint32_t)sizeof(stored_abort)) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(memcmp(&stored_abort, &aborted, sizeof(stored_abort)) == 0);
    REQUIRE(socketpair(AF_UNIX,
                       SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                       relay_sockets) == 0);
    REQUIRE(sagr_ccl_v1_carrier_session_send_abort(
                receiver, relay_sockets[0], &stored_abort) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 2U, 1U,
                                                &relay_receiver) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(sagr_ccl_v1_carrier_session_receive(
                relay_receiver, relay_sockets[1], &rank2,
                SAGR_CCL_V1_NO_CHUNK, 1U, &relayed_abort,
                (uint32_t)sizeof(relayed_abort)) ==
            SAGR_CCL_V1_STATUS_TIMED_OUT);
    REQUIRE(memcmp(&relayed_abort, &aborted, sizeof(relayed_abort)) == 0);
    REQUIRE(sagr_ccl_v1_carrier_session_get_abort(
                relay_receiver, &relayed_abort,
                (uint32_t)sizeof(relayed_abort)) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(memcmp(&relayed_abort, &aborted, sizeof(relayed_abort)) == 0);
    sagr_ccl_v1_carrier_session_destroy(&relay_receiver);
    REQUIRE(close(relay_sockets[0]) == 0 && close(relay_sockets[1]) == 0);
  }
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &rank1, 0U, 0U, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_ABORTED);
  sagr_ccl_v1_carrier_session_destroy(&sender);
  sagr_ccl_v1_carrier_session_destroy(&receiver);
  REQUIRE(sender == NULL && receiver == NULL);
  REQUIRE(count_open_descriptors() == descriptor_count_before);
  REQUIRE(close(sockets[0]) == 0 && close(sockets[1]) == 0);
  return 0;
}

static int test_session_replayed_data_aborts(void) {
  static const uint8_t payload[4] = {1U, 2U, 3U, 4U};
  sagr_ccl_v1_group_identity_t identity = make_identity(3U);
  sagr_ccl_v1_descriptor_t rank0 = make_descriptor(&identity, 0U);
  sagr_ccl_v1_descriptor_t rank1 = make_descriptor(&identity, 1U);
  sagr_ccl_v1_carrier_session_t sender = NULL;
  sagr_ccl_v1_carrier_session_t receiver = NULL;
  sagr_ccl_v1_carrier_record_t data;
  sagr_ccl_v1_carrier_record_t received;
  sagr_ccl_v1_carrier_record_t consumed;
  sagr_ccl_v1_carrier_session_info_t info;
  uint8_t output[sizeof(payload)] = {0U};
  int sockets[2] = {-1, -1};
  rank0.input_count = sizeof(payload) / sizeof(uint16_t) * identity.world_size;
  rank0.output_count = rank0.input_count;
  rank1.input_count = rank0.input_count;
  rank1.output_count = rank0.output_count;
  REQUIRE(socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     sockets) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 0U, 2U, &sender) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 1U, 2U, &receiver) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_prepare_data(
              sender, &rank0, 0U, payload, sizeof(payload), &data,
              (uint32_t)sizeof(data)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_send_data(sender, sockets[0], &data) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &rank1, 0U, 0U, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_consume(
              receiver, &rank1, 0U, &received, output, sizeof(output),
              &consumed, (uint32_t)sizeof(consumed)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_send_consumed(
              receiver, sockets[1], &consumed) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              sender, sockets[0], &rank0, 0U, 1U, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_SUCCESS);
  {
    int replay_fd = -1;
    uint32_t replay_crc = 0U;
    REQUIRE(sagr_ccl_v1_carrier_payload_create(
                payload, sizeof(payload), &replay_fd, &replay_crc) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(replay_crc == data.payload_crc32c);
    REQUIRE(sagr_ccl_v1_carrier_send(sockets[0], &data, replay_fd) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(close(replay_fd) == 0);
  }
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &rank1, 0U, 0U, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  REQUIRE(sagr_ccl_v1_carrier_session_info(
              receiver, &info, (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(info.phase == SAGR_CCL_V1_CARRIER_SESSION_ABORTED &&
          info.first_error == SAGR_CCL_V1_STATUS_OUT_OF_ORDER &&
          info.receiver_ready == 0U && info.receiver_consumed == 0U);
  sagr_ccl_v1_carrier_session_destroy(&sender);
  sagr_ccl_v1_carrier_session_destroy(&receiver);
  REQUIRE(close(sockets[0]) == 0 && close(sockets[1]) == 0);
  return 0;
}

static int test_session_multiple_credits_out_of_order(void) {
  static const uint8_t first_payload[4] = {1U, 3U, 5U, 7U};
  static const uint8_t second_payload[4] = {2U, 4U, 6U, 8U};
  sagr_ccl_v1_group_identity_t identity = make_identity(3U);
  sagr_ccl_v1_descriptor_t source_first = make_descriptor(&identity, 0U);
  sagr_ccl_v1_descriptor_t source_second = source_first;
  sagr_ccl_v1_descriptor_t destination_first;
  sagr_ccl_v1_descriptor_t destination_second;
  sagr_ccl_v1_carrier_session_t sender = NULL;
  sagr_ccl_v1_carrier_session_t receiver = NULL;
  sagr_ccl_v1_carrier_record_t data_first;
  sagr_ccl_v1_carrier_record_t data_second;
  sagr_ccl_v1_carrier_record_t received_second;
  sagr_ccl_v1_carrier_record_t received_first;
  sagr_ccl_v1_carrier_record_t consumed_second;
  sagr_ccl_v1_carrier_record_t consumed_first;
  sagr_ccl_v1_carrier_record_t ack;
  sagr_ccl_v1_carrier_session_info_t info;
  sagr_ccl_v1_status_t status;
  uint8_t output[4];
  int sockets[2] = {-1, -1};
  source_first.input_count = 6U;
  source_first.output_count = 6U;
  source_second = source_first;
  source_second.sequence = source_first.sequence + 1U;
  destination_first = source_first;
  destination_first.rank = 1U;
  destination_second = source_second;
  destination_second.rank = 1U;
  REQUIRE(socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     sockets) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 0U, 4U, &sender) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 1U, 4U, &receiver) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_prepare_data(
              sender, &source_first, 0U, first_payload,
              sizeof(first_payload), &data_first,
              (uint32_t)sizeof(data_first)) == SAGR_CCL_V1_STATUS_SUCCESS);
  status = sagr_ccl_v1_carrier_session_prepare_data(
      sender, &source_second, 0U, second_payload, sizeof(second_payload),
      &data_second, (uint32_t)sizeof(data_second));
  REQUIRE(status == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(data_first.slot_index == 0U && data_second.slot_index == 1U &&
          data_first.slot_generation < data_second.slot_generation);
  REQUIRE(sagr_ccl_v1_carrier_session_send_data(
              sender, sockets[0], &data_second) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_send_data(
              sender, sockets[0], &data_first) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &destination_second, 0U, 0U,
              &received_second, (uint32_t)sizeof(received_second)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &destination_first, 0U, 0U,
              &received_first, (uint32_t)sizeof(received_first)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_consume(
              receiver, &destination_second, 0U, &received_second, output,
              sizeof(output), &consumed_second,
              (uint32_t)sizeof(consumed_second)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(memcmp(output, second_payload, sizeof(output)) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_consume(
              receiver, &destination_first, 0U, &received_first, output,
              sizeof(output), &consumed_first,
              (uint32_t)sizeof(consumed_first)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(memcmp(output, first_payload, sizeof(output)) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_send_consumed(
              receiver, sockets[1], &consumed_first) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_send_consumed(
              receiver, sockets[1], &consumed_second) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              sender, sockets[0], &source_first, 0U, 1U, &ack,
              (uint32_t)sizeof(ack)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              sender, sockets[0], &source_second, 0U, 1U, &ack,
              (uint32_t)sizeof(ack)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_info(
              sender, &info, (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(info.sender_inflight == 0U &&
          info.phase == SAGR_CCL_V1_CARRIER_SESSION_RUNNING);
  sagr_ccl_v1_carrier_session_destroy(&sender);
  sagr_ccl_v1_carrier_session_destroy(&receiver);
  REQUIRE(close(sockets[0]) == 0 && close(sockets[1]) == 0);
  return 0;
}

static int test_session_zero_payload_barrier(void) {
  sagr_ccl_v1_group_identity_t identity = make_identity(5U);
  sagr_ccl_v1_descriptor_t source = make_operation_descriptor(
      &identity, 0U, SAGR_CCL_V1_OPERATION_BARRIER);
  sagr_ccl_v1_plan_step_t steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
  sagr_ccl_v1_descriptor_t destination;
  sagr_ccl_v1_carrier_session_t sender = NULL;
  sagr_ccl_v1_carrier_session_t receiver = NULL;
  sagr_ccl_v1_carrier_record_t data;
  sagr_ccl_v1_carrier_record_t received;
  sagr_ccl_v1_carrier_record_t consumed;
  uint32_t step_count = 0U;
  int sockets[2] = {-1, -1};
  REQUIRE(sagr_ccl_v1_plan_rank(&source, steps,
                                SAGR_CCL_V1_MAX_PLAN_STEPS,
                                &step_count) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(step_count == 3U && steps[0].send_count_elements == 0U);
  destination = source;
  destination.rank = steps[0].send_rank;
  REQUIRE(socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     sockets) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, source.rank, 1U,
                                              &sender) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, destination.rank, 1U,
                                              &receiver) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_prepare_data(
              sender, &source, 0U, NULL, 0U, &data,
              (uint32_t)sizeof(data)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(data.payload_bytes == 0U && data.payload_crc32c == 0U);
  REQUIRE(sagr_ccl_v1_carrier_session_send_data(sender, sockets[0], &data) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &destination, 0U, source.rank, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_consume(
              receiver, &destination, 0U, &received, NULL, 0U, &consumed,
              (uint32_t)sizeof(consumed)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_send_consumed(
              receiver, sockets[1], &consumed) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              sender, sockets[0], &source, 0U, destination.rank, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_SUCCESS);
  sagr_ccl_v1_carrier_session_destroy(&sender);
  sagr_ccl_v1_carrier_session_destroy(&receiver);
  REQUIRE(close(sockets[0]) == 0 && close(sockets[1]) == 0);
  return 0;
}

static int test_session_automatic_abort_and_descriptor_preflight(void) {
  static const uint8_t payload[4] = {9U, 8U, 7U, 6U};
  sagr_ccl_v1_group_identity_t identity = make_identity(3U);
  sagr_ccl_v1_descriptor_t rank0 = make_descriptor(&identity, 0U);
  sagr_ccl_v1_descriptor_t rank1 = make_descriptor(&identity, 1U);
  sagr_ccl_v1_descriptor_t invalid_rank1;
  sagr_ccl_v1_carrier_session_t sender = NULL;
  sagr_ccl_v1_carrier_session_t receiver = NULL;
  sagr_ccl_v1_carrier_record_t data;
  sagr_ccl_v1_carrier_record_t received;
  sagr_ccl_v1_carrier_record_t automatic_abort;
  int sockets[2] = {-1, -1};
  rank0.input_count = 6U;
  rank0.output_count = 6U;
  rank1.input_count = 6U;
  rank1.output_count = 6U;
  REQUIRE(socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     sockets) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 0U, 1U, &sender) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 1U, 1U, &receiver) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_prepare_data(
              sender, &rank0, 0U, payload, sizeof(payload), &data,
              (uint32_t)sizeof(data)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_send_data(sender, sockets[0], &data) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  invalid_rank1 = rank1;
  invalid_rank1.reserved[0] = UINT8_C(1);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &invalid_rank1, 0U, 0U, &received,
              (uint32_t)sizeof(received)) ==
          SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &rank1, 0U, 0U, &received,
              (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_SUCCESS);
  sagr_ccl_v1_carrier_session_destroy(&receiver);
  REQUIRE(close(sockets[1]) == 0);
  sockets[1] = -1;
  sagr_ccl_v1_carrier_session_destroy(&sender);
  REQUIRE(close(sockets[0]) == 0);
  sockets[0] = -1;
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 0U, 1U, &sender) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     sockets) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_prepare_data(
              sender, &rank0, 0U, payload, sizeof(payload), &data,
              (uint32_t)sizeof(data)) == SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(close(sockets[1]) == 0);
  sockets[1] = -1;
  REQUIRE(sagr_ccl_v1_carrier_session_send_data(sender, sockets[0], &data) ==
          SAGR_CCL_V1_STATUS_PEER_LOST);
  REQUIRE(sagr_ccl_v1_carrier_session_get_abort(
              sender, &automatic_abort,
              (uint32_t)sizeof(automatic_abort)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(automatic_abort.kind == SAGR_CCL_V1_CARRIER_MESSAGE_ABORT &&
          automatic_abort.source_rank == 0U &&
          automatic_abort.failed_rank == data.destination_rank &&
          automatic_abort.status == SAGR_CCL_V1_STATUS_PEER_LOST &&
          automatic_abort.sequence == data.sequence &&
          memcmp(automatic_abort.descriptor_sha256, data.descriptor_sha256,
                 SAGR_CCL_V1_SHA256_BYTES) == 0);
  sagr_ccl_v1_carrier_session_destroy(&sender);
  REQUIRE(close(sockets[0]) == 0);
  return 0;
}

static int test_session_version_mismatch_abort_relay(void) {
  static const uint8_t payload[4] = {0x61U, 0x62U, 0x63U, 0x64U};
  sagr_ccl_v1_group_identity_t identity = make_identity(3U);
  sagr_ccl_v1_descriptor_t rank0 = make_descriptor(&identity, 0U);
  sagr_ccl_v1_descriptor_t rank1 = make_descriptor(&identity, 1U);
  sagr_ccl_v1_descriptor_t rank2 = make_descriptor(&identity, 2U);
  sagr_ccl_v1_carrier_session_t receiver = NULL;
  sagr_ccl_v1_carrier_session_t relay_receiver = NULL;
  sagr_ccl_v1_carrier_record_t data;
  sagr_ccl_v1_carrier_record_t received;
  sagr_ccl_v1_carrier_record_t stored_abort;
  uint8_t wire[SAGR_CCL_V1_CARRIER_WIRE_BYTES];
  int sockets[2] = {-1, -1};
  int relay_sockets[2] = {-1, -1};
  int payload_fd = -1;
  uint32_t payload_crc = 0U;
  rank0.input_count = 6U;
  rank0.output_count = 6U;
  rank1.input_count = 6U;
  rank1.output_count = 6U;
  rank2.input_count = 6U;
  rank2.output_count = 6U;
  REQUIRE(sagr_ccl_v1_carrier_payload_create(
              payload, sizeof(payload), &payload_fd, &payload_crc) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_record_from_plan(
              &rank0, 0U, 0U, SAGR_CCL_V1_CARRIER_MESSAGE_DATA, 0U, 1U,
              payload_crc, &data, (uint32_t)sizeof(data)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(data.payload_bytes == sizeof(payload));
  REQUIRE(sagr_ccl_v1_carrier_record_encode(&data, wire, sizeof(wire)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  wire[5] ^= UINT8_C(1);
  REQUIRE(socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     sockets) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 1U, 1U, &receiver) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(send_raw(sockets[0], wire, &payload_fd, 1U) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              receiver, sockets[1], &rank1, 0U, 0U, &received,
              (uint32_t)sizeof(received)) ==
          SAGR_CCL_V1_STATUS_VERSION_MISMATCH);
  REQUIRE(sagr_ccl_v1_carrier_session_get_abort(
              receiver, &stored_abort, (uint32_t)sizeof(stored_abort)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(stored_abort.status == SAGR_CCL_V1_STATUS_VERSION_MISMATCH &&
          stored_abort.source_rank == 1U && stored_abort.failed_rank == 0U &&
          stored_abort.sequence == rank1.sequence);
  REQUIRE(socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     relay_sockets) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, 2U, 1U,
                                              &relay_receiver) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_send_abort(
              receiver, relay_sockets[0], &stored_abort) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_carrier_session_receive(
              relay_receiver, relay_sockets[1], &rank2,
              SAGR_CCL_V1_NO_CHUNK, 1U, &received,
              (uint32_t)sizeof(received)) ==
          SAGR_CCL_V1_STATUS_VERSION_MISMATCH);
  REQUIRE(memcmp(&received, &stored_abort, sizeof(received)) == 0);
  REQUIRE(sagr_ccl_v1_carrier_session_get_abort(
              relay_receiver, &received, (uint32_t)sizeof(received)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(memcmp(&received, &stored_abort, sizeof(received)) == 0);
  sagr_ccl_v1_carrier_session_destroy(&receiver);
  sagr_ccl_v1_carrier_session_destroy(&relay_receiver);
  REQUIRE(close(sockets[0]) == 0 && close(sockets[1]) == 0 &&
          close(relay_sockets[0]) == 0 && close(relay_sockets[1]) == 0 &&
          close(payload_fd) == 0);
  return 0;
}

static int test_session_world2_16_roundtrip(void) {
  static const uint8_t payload[32] = {
      0x31U, 0x32U, 0x33U, 0x34U, 0x35U, 0x36U, 0x37U, 0x38U,
      0x39U, 0x3aU, 0x3bU, 0x3cU, 0x3dU, 0x3eU, 0x3fU, 0x40U,
      0x41U, 0x42U, 0x43U, 0x44U, 0x45U, 0x46U, 0x47U, 0x48U,
      0x49U, 0x4aU, 0x4bU, 0x4cU, 0x4dU, 0x4eU, 0x4fU, 0x50U};
  uint32_t world_size;
  for (world_size = 2U; world_size <= 16U; ++world_size) {
    uint32_t operation;
    for (operation = SAGR_CCL_V1_OPERATION_ALL_REDUCE;
         operation <= SAGR_CCL_V1_OPERATION_BARRIER; ++operation) {
      sagr_ccl_v1_group_identity_t identity = make_identity(world_size);
      sagr_ccl_v1_descriptor_t source;
      sagr_ccl_v1_descriptor_t destination;
      sagr_ccl_v1_plan_step_t steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
      sagr_ccl_v1_carrier_session_t sender = NULL;
      sagr_ccl_v1_carrier_session_t receiver = NULL;
      sagr_ccl_v1_carrier_record_t data;
      sagr_ccl_v1_carrier_record_t received;
      sagr_ccl_v1_carrier_record_t consumed;
      uint8_t output[sizeof(payload)] = {0U};
      uint64_t payload_bytes;
      uint32_t rank;
      uint32_t step = 0U;
      uint32_t step_count = 0U;
      int found = 0;
      int sockets[2] = {-1, -1};
      for (rank = 0U; rank < world_size && !found; ++rank) {
        source = make_operation_descriptor(&identity, rank, operation);
        REQUIRE(sagr_ccl_v1_plan_rank(
                    &source, steps, SAGR_CCL_V1_MAX_PLAN_STEPS,
                    &step_count) == SAGR_CCL_V1_STATUS_SUCCESS);
        for (step = 0U; step < step_count; ++step) {
          if (steps[step].action == SAGR_CCL_V1_PLAN_ACTION_SEND ||
              steps[step].action == SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE) {
            found = 1;
            break;
          }
        }
      }
      REQUIRE(found);
      destination = source;
      destination.rank = steps[step].send_rank;
      payload_bytes = steps[step].send_count_elements *
                      (source.dtype == SAGR_CCL_V1_DTYPE_BF16
                           ? UINT64_C(2)
                           : source.dtype == SAGR_CCL_V1_DTYPE_UINT8
                                 ? UINT64_C(1)
                                 : source.dtype == SAGR_CCL_V1_DTYPE_NONE
                                       ? UINT64_C(0)
                                       : UINT64_C(4));
      REQUIRE(payload_bytes <= sizeof(payload));
      REQUIRE(socketpair(AF_UNIX,
                         SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                         sockets) == 0);
      REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, source.rank, 2U,
                                                  &sender) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, destination.rank,
                                                  2U, &receiver) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(sagr_ccl_v1_carrier_session_prepare_data(
                  sender, &source, step,
                  payload_bytes == 0U ? NULL : payload, payload_bytes, &data,
                  (uint32_t)sizeof(data)) == SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(sagr_ccl_v1_carrier_session_send_data(
                  sender, sockets[0], &data) == SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(sagr_ccl_v1_carrier_session_receive(
                  receiver, sockets[1], &destination, step, source.rank,
                  &received, (uint32_t)sizeof(received)) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(sagr_ccl_v1_carrier_session_consume(
                  receiver, &destination, step, &received,
                  payload_bytes == 0U ? NULL : output, payload_bytes, &consumed,
                  (uint32_t)sizeof(consumed)) == SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(payload_bytes == 0U ||
              memcmp(output, payload, (size_t)payload_bytes) == 0);
      REQUIRE(sagr_ccl_v1_carrier_session_send_consumed(
                  receiver, sockets[1], &consumed) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(sagr_ccl_v1_carrier_session_receive(
                  sender, sockets[0], &source, step, destination.rank, &received,
                  (uint32_t)sizeof(received)) == SAGR_CCL_V1_STATUS_SUCCESS);
      sagr_ccl_v1_carrier_session_destroy(&sender);
      sagr_ccl_v1_carrier_session_destroy(&receiver);
      REQUIRE(close(sockets[0]) == 0 && close(sockets[1]) == 0);
    }
  }
  return 0;
}

static int test_session_world16_max_inflight_abort(void) {
  static const uint8_t payload[3] = {0x41U, 0x42U, 0x43U};
  sagr_ccl_v1_group_identity_t identity = make_identity(16U);
  sagr_ccl_v1_carrier_session_t sessions[16] = {NULL};
  sagr_ccl_v1_carrier_record_t data;
  sagr_ccl_v1_carrier_record_t aborted;
  sagr_ccl_v1_carrier_session_info_t info;
  uint32_t rank;
  uint32_t slot;
  int descriptor_count_before = count_open_descriptors();
  REQUIRE(descriptor_count_before >= 0);
  for (rank = 0U; rank < identity.world_size; ++rank) {
    sagr_ccl_v1_descriptor_t source = make_operation_descriptor(
        &identity, rank, SAGR_CCL_V1_OPERATION_ALL_GATHER);
    REQUIRE(sagr_ccl_v1_carrier_session_create(&identity, rank, 16U,
                                                &sessions[rank]) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    for (slot = 0U; slot < 16U; ++slot) {
      REQUIRE(sagr_ccl_v1_carrier_session_prepare_data(
                  sessions[rank], &source, 0U, payload, sizeof(payload), &data,
                  (uint32_t)sizeof(data)) == SAGR_CCL_V1_STATUS_SUCCESS);
    }
    REQUIRE(sagr_ccl_v1_carrier_session_info(
                sessions[rank], &info, (uint32_t)sizeof(info)) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(info.sender_inflight == 16U);
  }
  REQUIRE(count_open_descriptors() == descriptor_count_before + 256);
  for (rank = 0U; rank < identity.world_size; ++rank) {
    sagr_ccl_v1_descriptor_t source = make_operation_descriptor(
        &identity, rank, SAGR_CCL_V1_OPERATION_ALL_GATHER);
    REQUIRE(sagr_ccl_v1_carrier_session_abort(
                sessions[rank], &source, rank,
                SAGR_CCL_V1_STATUS_TIMED_OUT, &aborted,
                (uint32_t)sizeof(aborted)) == SAGR_CCL_V1_STATUS_TIMED_OUT);
  }
  REQUIRE(count_open_descriptors() == descriptor_count_before);
  for (rank = 0U; rank < identity.world_size; ++rank) {
    sagr_ccl_v1_carrier_session_destroy(&sessions[rank]);
  }
  REQUIRE(count_open_descriptors() == descriptor_count_before);
  return 0;
}

int main(void) {
  REQUIRE(test_record_codec_and_descriptor_binding() == 0);
  REQUIRE(test_all_world_plan_record_symmetry() == 0);
  REQUIRE(test_plan_record_mutation_matrix() == 0);
  REQUIRE(test_credit_depths() == 0);
  REQUIRE(test_credit_matrix() == 0);
  REQUIRE(test_world16_simultaneous_credit_capacity() == 0);
  REQUIRE(test_slot_lifecycle_and_faults() == 0);
  REQUIRE(test_payload_and_seqpacket_transport() == 0);
  REQUIRE(test_session_lifecycle_and_abort() == 0);
  REQUIRE(test_session_replayed_data_aborts() == 0);
  REQUIRE(test_session_multiple_credits_out_of_order() == 0);
  REQUIRE(test_session_zero_payload_barrier() == 0);
  REQUIRE(test_session_automatic_abort_and_descriptor_preflight() == 0);
  REQUIRE(test_session_version_mismatch_abort_relay() == 0);
  REQUIRE(test_session_world2_16_roundtrip() == 0);
  REQUIRE(test_session_world16_max_inflight_abort() == 0);
  puts("ccl_carrier_v1: world2-16 codec/credit/session/fault tests passed");
  return 0;
}
