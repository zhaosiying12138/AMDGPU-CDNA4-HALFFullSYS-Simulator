/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/ccl_v1.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

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
  identity.group_generation = UINT64_C(7);
  for (index = 0; index < SAGR_CCL_V1_UUID_BYTES; ++index) {
    identity.job_uuid[index] = (uint8_t)(index + 1U);
    identity.group_uuid[index] = (uint8_t)(0x40U + index);
  }
  for (index = 0; index < SAGR_CCL_V1_SHA256_BYTES; ++index) {
    identity.model_identity_sha256[index] = (uint8_t)(0x80U + index);
  }
  return identity;
}

static sagr_ccl_v1_descriptor_t make_descriptor(
    uint32_t world_size, uint32_t rank, uint32_t operation, uint32_t dtype,
    uint32_t reduction, uint64_t input_count, uint64_t output_count) {
  sagr_ccl_v1_descriptor_t descriptor;
  (void)sagr_ccl_v1_descriptor_init(&descriptor,
                                    (uint32_t)sizeof(descriptor));
  descriptor.group = make_identity(world_size);
  descriptor.sequence = 1U;
  descriptor.input_count = input_count;
  descriptor.output_count = output_count;
  descriptor.rank = rank;
  descriptor.operation = operation;
  descriptor.reduction = reduction;
  descriptor.dtype = dtype;
  descriptor.root_rank = SAGR_CCL_V1_NO_RANK;
  return descriptor;
}

static int test_oversized_init_does_not_touch_trailing_memory(void) {
  struct identity_with_canary {
    sagr_ccl_v1_group_identity_t value;
    uint64_t canary;
  } identity;
  struct descriptor_with_canary {
    sagr_ccl_v1_descriptor_t value;
    uint64_t canary;
  } descriptor;
  struct state_with_canary {
    sagr_ccl_v1_group_state_t value;
    uint64_t canary;
  } state;
  struct snapshot_with_canary {
    sagr_ccl_v1_group_snapshot_t value;
    uint64_t canary;
  } snapshot;
  const uint64_t canary = UINT64_C(0x5aa55aa5c33cc33c);
  sagr_ccl_v1_group_identity_t valid = make_identity(3U);
  identity.canary = canary;
  descriptor.canary = canary;
  state.canary = canary;
  snapshot.canary = canary;
  REQUIRE(sagr_ccl_v1_group_identity_init(
              &identity.value, (uint32_t)sizeof(identity)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(identity.canary == canary);
  REQUIRE(sagr_ccl_v1_descriptor_init(
              &descriptor.value, (uint32_t)sizeof(descriptor)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(descriptor.canary == canary);
  REQUIRE(sagr_ccl_v1_group_state_init(&state.value, (uint32_t)sizeof(state),
                                       &valid) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(state.canary == canary);
  REQUIRE(sagr_ccl_v1_group_state_snapshot(
              &state.value, &snapshot.value, (uint32_t)sizeof(snapshot)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(snapshot.canary == canary);
  return 0;
}

static int test_identity_and_descriptor_legality(void) {
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    sagr_ccl_v1_group_identity_t identity = make_identity(world);
    REQUIRE(sagr_ccl_v1_group_identity_validate(&identity) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
  }
  {
    sagr_ccl_v1_group_identity_t invalid = make_identity(1U);
    REQUIRE(sagr_ccl_v1_group_identity_validate(&invalid) ==
            SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
    invalid = make_identity(5U);
    REQUIRE(sagr_ccl_v1_group_identity_validate(&invalid) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    invalid = make_identity(SAGR_CCL_V1_MAX_WORLD_SIZE + 1U);
    REQUIRE(sagr_ccl_v1_group_identity_validate(&invalid) ==
            SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
    invalid = make_identity(3U);
    invalid.epoch = 0U;
    REQUIRE(sagr_ccl_v1_group_identity_validate(&invalid) ==
            SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
    invalid = make_identity(3U);
    memset(invalid.model_identity_sha256, 0,
           sizeof(invalid.model_identity_sha256));
    REQUIRE(sagr_ccl_v1_group_identity_validate(&invalid) ==
            SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  }
  {
    sagr_ccl_v1_descriptor_t all_reduce = make_descriptor(
        3U, 0U, SAGR_CCL_V1_OPERATION_ALL_REDUCE,
        SAGR_CCL_V1_DTYPE_BF16, SAGR_CCL_V1_REDUCTION_SUM, 10U, 10U);
    sagr_ccl_v1_descriptor_t all_gather = make_descriptor(
        3U, 1U, SAGR_CCL_V1_OPERATION_ALL_GATHER,
        SAGR_CCL_V1_DTYPE_UINT32, SAGR_CCL_V1_REDUCTION_NONE, 7U, 21U);
    sagr_ccl_v1_descriptor_t reduce_scatter = make_descriptor(
        3U, 2U, SAGR_CCL_V1_OPERATION_REDUCE_SCATTER,
        SAGR_CCL_V1_DTYPE_FP32, SAGR_CCL_V1_REDUCTION_SUM, 21U, 7U);
    sagr_ccl_v1_descriptor_t broadcast = make_descriptor(
        3U, 1U, SAGR_CCL_V1_OPERATION_BROADCAST,
        SAGR_CCL_V1_DTYPE_INT32, SAGR_CCL_V1_REDUCTION_NONE, 9U, 9U);
    sagr_ccl_v1_descriptor_t barrier = make_descriptor(
        3U, 1U, SAGR_CCL_V1_OPERATION_BARRIER, SAGR_CCL_V1_DTYPE_NONE,
        SAGR_CCL_V1_REDUCTION_NONE, 0U, 0U);
    broadcast.root_rank = 2U;
    REQUIRE(sagr_ccl_v1_descriptor_validate(&all_reduce) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(sagr_ccl_v1_descriptor_validate(&all_gather) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(sagr_ccl_v1_descriptor_validate(&reduce_scatter) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(sagr_ccl_v1_descriptor_validate(&broadcast) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(sagr_ccl_v1_descriptor_validate(&barrier) ==
            SAGR_CCL_V1_STATUS_SUCCESS);

    all_reduce.dtype = SAGR_CCL_V1_DTYPE_INT32;
    REQUIRE(sagr_ccl_v1_descriptor_validate(&all_reduce) ==
            SAGR_CCL_V1_STATUS_NOT_SUPPORTED);
    reduce_scatter.reduction = SAGR_CCL_V1_REDUCTION_NONE;
    REQUIRE(sagr_ccl_v1_descriptor_validate(&reduce_scatter) ==
            SAGR_CCL_V1_STATUS_NOT_SUPPORTED);
    all_gather.output_count = 20U;
    REQUIRE(sagr_ccl_v1_descriptor_validate(&all_gather) ==
            SAGR_CCL_V1_STATUS_INVALID_ARGUMENT);
    barrier.dtype = SAGR_CCL_V1_DTYPE_UINT8;
    REQUIRE(sagr_ccl_v1_descriptor_validate(&barrier) ==
            SAGR_CCL_V1_STATUS_INVALID_ARGUMENT);
  }
  REQUIRE(strcmp(sagr_ccl_v1_status_string(SAGR_CCL_V1_STATUS_PEER_LOST),
                 "peer lost") == 0);
  REQUIRE(strcmp(sagr_ccl_v1_status_string(999), "unknown status") == 0);
  return 0;
}

static int test_codec_roundtrip_and_corruption(void) {
  sagr_ccl_v1_descriptor_t source = make_descriptor(
      3U, 2U, SAGR_CCL_V1_OPERATION_ALL_REDUCE, SAGR_CCL_V1_DTYPE_FP32,
      SAGR_CCL_V1_REDUCTION_SUM, 1001U, 1001U);
  sagr_ccl_v1_descriptor_t decoded;
  uint8_t wire[SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES];
  uint8_t corrupted[SAGR_CCL_V1_DESCRIPTOR_WIRE_BYTES];
  REQUIRE(sagr_ccl_v1_descriptor_encode(&source, wire, sizeof(wire)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_descriptor_decode(wire, sizeof(wire), &decoded,
                                        (uint32_t)sizeof(decoded)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(decoded.sequence == source.sequence &&
          decoded.input_count == source.input_count &&
          decoded.output_count == source.output_count &&
          decoded.rank == source.rank && decoded.operation == source.operation &&
          decoded.dtype == source.dtype &&
          sagr_ccl_v1_group_identity_equal(&decoded.group, &source.group));

  memcpy(corrupted, wire, sizeof(corrupted));
  corrupted[120] ^= UINT8_C(1);
  REQUIRE(sagr_ccl_v1_descriptor_decode(corrupted, sizeof(corrupted), &decoded,
                                        (uint32_t)sizeof(decoded)) ==
          SAGR_CCL_V1_STATUS_CHECKSUM_ERROR);
  memcpy(corrupted, wire, sizeof(corrupted));
  corrupted[148] = 1U;
  REQUIRE(sagr_ccl_v1_descriptor_decode(corrupted, sizeof(corrupted), &decoded,
                                        (uint32_t)sizeof(decoded)) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  memcpy(corrupted, wire, sizeof(corrupted));
  corrupted[5] = 2U;
  REQUIRE(sagr_ccl_v1_descriptor_decode(corrupted, sizeof(corrupted), &decoded,
                                        (uint32_t)sizeof(decoded)) ==
          SAGR_CCL_V1_STATUS_VERSION_MISMATCH);
  REQUIRE(sagr_ccl_v1_descriptor_decode(wire, sizeof(wire) - 1U, &decoded,
                                        (uint32_t)sizeof(decoded)) ==
          SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL);
  return 0;
}

static int test_all_reduce_planner_world(uint32_t world) {
  sagr_ccl_v1_plan_step_t
      plans[SAGR_CCL_V1_MAX_WORLD_SIZE][SAGR_CCL_V1_MAX_PLAN_STEPS];
  uint32_t counts[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t rank;
  for (rank = 0U; rank < world; ++rank) {
    sagr_ccl_v1_descriptor_t descriptor = make_descriptor(
        world, rank, SAGR_CCL_V1_OPERATION_ALL_REDUCE,
        SAGR_CCL_V1_DTYPE_BF16, SAGR_CCL_V1_REDUCTION_SUM, 1001U, 1001U);
    sagr_ccl_v1_plan_step_t *steps = plans[rank];
    uint32_t index;
    uint32_t gather_chunks = 0U;
    counts[rank] = 0U;
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps,
                                  SAGR_CCL_V1_MAX_PLAN_STEPS,
                                  &counts[rank]) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(counts[rank] == 2U * (world - 1U));
    for (index = 0U; index < world - 1U; ++index) {
      const sagr_ccl_v1_plan_step_t *reduce = &steps[index];
      const sagr_ccl_v1_plan_step_t *gather = &steps[world - 1U + index];
      REQUIRE(reduce->phase == SAGR_CCL_V1_PLAN_PHASE_REDUCE_SCATTER);
      REQUIRE(gather->phase == SAGR_CCL_V1_PLAN_PHASE_ALL_GATHER);
      REQUIRE(reduce->action == SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE &&
              gather->action == SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE);
      REQUIRE(reduce->send_rank == (rank + 1U) % world &&
              reduce->receive_rank == (rank + world - 1U) % world);
      REQUIRE(reduce->send_rank != rank && reduce->receive_rank != rank);
      REQUIRE(reduce->send_count_elements > 0U &&
              reduce->receive_count_elements > 0U);
      REQUIRE(gather->receive_chunk != rank);
      REQUIRE((gather_chunks & (UINT32_C(1) << gather->receive_chunk)) == 0U);
      gather_chunks |= UINT32_C(1) << gather->receive_chunk;
    }
    REQUIRE(gather_chunks == (((UINT32_C(1) << world) - 1U) &
                              ~(UINT32_C(1) << rank)));
  }
  for (rank = 0U; rank < world; ++rank) {
    uint32_t index;
    for (index = 0U; index < counts[rank]; ++index) {
      const sagr_ccl_v1_plan_step_t *send = &plans[rank][index];
      const sagr_ccl_v1_plan_step_t *receive =
          &plans[send->send_rank][index];
      REQUIRE(receive->receive_rank == rank &&
              receive->receive_chunk == send->send_chunk &&
              receive->receive_offset_elements == send->send_offset_elements &&
              receive->receive_count_elements == send->send_count_elements &&
              receive->phase == send->phase &&
              receive->step_index == send->step_index);
    }
  }
  return 0;
}

static int test_planner_peer_symmetry(sagr_ccl_v1_descriptor_t descriptor) {
  sagr_ccl_v1_plan_step_t
      plans[SAGR_CCL_V1_MAX_WORLD_SIZE][SAGR_CCL_V1_MAX_PLAN_STEPS];
  uint32_t counts[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t rank;
  for (rank = 0U; rank < descriptor.group.world_size; ++rank) {
    descriptor.rank = rank;
    counts[rank] = 0U;
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, plans[rank],
                                  SAGR_CCL_V1_MAX_PLAN_STEPS,
                                  &counts[rank]) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
  }
  for (rank = 0U; rank < descriptor.group.world_size; ++rank) {
    uint32_t index;
    for (index = 0U; index < counts[rank]; ++index) {
      const sagr_ccl_v1_plan_step_t *step = &plans[rank][index];
      if (step->action == SAGR_CCL_V1_PLAN_ACTION_SEND ||
          step->action == SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE) {
        const sagr_ccl_v1_plan_step_t *peer =
            &plans[step->send_rank][index];
        REQUIRE(peer->action == SAGR_CCL_V1_PLAN_ACTION_RECEIVE ||
                peer->action == SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE);
        REQUIRE(peer->receive_rank == rank && peer->phase == step->phase &&
                peer->receive_chunk == step->send_chunk &&
                peer->receive_offset_elements == step->send_offset_elements &&
                peer->receive_count_elements == step->send_count_elements);
      }
      if (step->action == SAGR_CCL_V1_PLAN_ACTION_RECEIVE ||
          step->action == SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE) {
        const sagr_ccl_v1_plan_step_t *peer =
            &plans[step->receive_rank][index];
        REQUIRE(peer->action == SAGR_CCL_V1_PLAN_ACTION_SEND ||
                peer->action == SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE);
        REQUIRE(peer->send_rank == rank && peer->phase == step->phase &&
                peer->send_chunk == step->receive_chunk &&
                peer->send_offset_elements == step->receive_offset_elements &&
                peer->send_count_elements == step->receive_count_elements);
      }
    }
  }
  return 0;
}

static int test_planner_matrix(void) {
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    sagr_ccl_v1_descriptor_t descriptor;
    sagr_ccl_v1_plan_step_t steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
    uint32_t count;
    REQUIRE(test_all_reduce_planner_world(world) == 0);

    descriptor = make_descriptor(
        world, world - 1U, SAGR_CCL_V1_OPERATION_ALL_GATHER,
        SAGR_CCL_V1_DTYPE_UINT8, SAGR_CCL_V1_REDUCTION_NONE, 5U,
        (uint64_t)world * 5U);
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps,
                                  SAGR_CCL_V1_MAX_PLAN_STEPS, &count) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(count == world - 1U);
    REQUIRE(test_planner_peer_symmetry(descriptor) == 0);

    descriptor = make_descriptor(
        world, world - 1U, SAGR_CCL_V1_OPERATION_REDUCE_SCATTER,
        SAGR_CCL_V1_DTYPE_FP32, SAGR_CCL_V1_REDUCTION_SUM,
        (uint64_t)world * 5U, 5U);
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps,
                                  SAGR_CCL_V1_MAX_PLAN_STEPS, &count) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(count == world - 1U);
    REQUIRE(test_planner_peer_symmetry(descriptor) == 0);

    descriptor = make_descriptor(
        world, world - 1U, SAGR_CCL_V1_OPERATION_BROADCAST,
        SAGR_CCL_V1_DTYPE_UINT32, SAGR_CCL_V1_REDUCTION_NONE, 5U, 5U);
    descriptor.root_rank = 0U;
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps,
                                  SAGR_CCL_V1_MAX_PLAN_STEPS, &count) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(count == world - 1U);
    REQUIRE(test_planner_peer_symmetry(descriptor) == 0);

    descriptor = make_descriptor(
        world, world - 1U, SAGR_CCL_V1_OPERATION_BARRIER,
        SAGR_CCL_V1_DTYPE_NONE, SAGR_CCL_V1_REDUCTION_NONE, 0U, 0U);
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps,
                                  SAGR_CCL_V1_MAX_PLAN_STEPS, &count) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(count >= 1U && count <= 4U);
    REQUIRE(test_planner_peer_symmetry(descriptor) == 0);
  }
  {
    sagr_ccl_v1_descriptor_t descriptor = make_descriptor(
        3U, 0U, SAGR_CCL_V1_OPERATION_ALL_REDUCE,
        SAGR_CCL_V1_DTYPE_BF16, SAGR_CCL_V1_REDUCTION_SUM, 10U, 10U);
    sagr_ccl_v1_plan_step_t steps[4];
    uint32_t count;
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps, 4U, &count) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(count == 4U);
    REQUIRE(steps[0].send_chunk == 2U && steps[0].receive_chunk == 1U);
    REQUIRE(steps[1].send_chunk == 1U && steps[1].receive_chunk == 0U);
    REQUIRE(steps[2].send_chunk == 0U && steps[2].receive_chunk == 2U);
    REQUIRE(steps[3].send_chunk == 2U && steps[3].receive_chunk == 1U);
    REQUIRE(steps[0].send_count_elements == 3U &&
            steps[0].receive_count_elements == 3U);
    REQUIRE(steps[1].receive_count_elements == 4U);
  }
  {
    sagr_ccl_v1_descriptor_t descriptor = make_descriptor(
        3U, 1U, SAGR_CCL_V1_OPERATION_ALL_GATHER,
        SAGR_CCL_V1_DTYPE_UINT32, SAGR_CCL_V1_REDUCTION_NONE, 7U, 21U);
    sagr_ccl_v1_plan_step_t steps[2];
    uint32_t count;
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps, 1U, &count) ==
            SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL);
    REQUIRE(count == 2U);
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps, 2U, &count) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(steps[0].send_count_elements == 7U &&
            steps[0].receive_count_elements == 7U);
  }
  {
    uint32_t rank;
    for (rank = 0U; rank < 3U; ++rank) {
      sagr_ccl_v1_descriptor_t descriptor = make_descriptor(
          3U, rank, SAGR_CCL_V1_OPERATION_BROADCAST,
          SAGR_CCL_V1_DTYPE_INT32, SAGR_CCL_V1_REDUCTION_NONE, 9U, 9U);
      sagr_ccl_v1_plan_step_t steps[2];
      uint32_t count;
      descriptor.root_rank = 2U;
      REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps, 2U, &count) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(count == 2U);
      if (rank == 2U) {
        REQUIRE(steps[0].action == SAGR_CCL_V1_PLAN_ACTION_SEND &&
                steps[0].send_rank == 0U);
      } else if (rank == 0U) {
        REQUIRE(steps[0].action == SAGR_CCL_V1_PLAN_ACTION_RECEIVE &&
                steps[1].action == SAGR_CCL_V1_PLAN_ACTION_SEND);
      } else {
        REQUIRE(steps[1].action == SAGR_CCL_V1_PLAN_ACTION_RECEIVE);
      }
    }
  }
  {
    sagr_ccl_v1_descriptor_t descriptor = make_descriptor(
        3U, 0U, SAGR_CCL_V1_OPERATION_BARRIER, SAGR_CCL_V1_DTYPE_NONE,
        SAGR_CCL_V1_REDUCTION_NONE, 0U, 0U);
    sagr_ccl_v1_plan_step_t steps[2];
    uint32_t count;
    REQUIRE(sagr_ccl_v1_plan_rank(&descriptor, steps, 2U, &count) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(count == 2U);
    REQUIRE(steps[0].send_rank == 1U && steps[0].receive_rank == 2U);
    REQUIRE(steps[1].send_rank == 2U && steps[1].receive_rank == 1U);
  }
  return 0;
}

static int join_all(sagr_ccl_v1_group_state_t *state,
                    const sagr_ccl_v1_group_identity_t *identity) {
  uint32_t rank;
  for (rank = 0U; rank < identity->world_size; ++rank) {
    REQUIRE(sagr_ccl_v1_group_state_join(state, identity, rank) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
  }
  return 0;
}

static int test_state_machine_success(void) {
  sagr_ccl_v1_group_identity_t identity = make_identity(3U);
  sagr_ccl_v1_group_state_t state;
  sagr_ccl_v1_group_snapshot_t snapshot;
  uint32_t rank;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_JOINING);
  REQUIRE(join_all(&state, &identity) == 0);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_READY);
  for (rank = 0U; rank < identity.world_size; ++rank) {
    sagr_ccl_v1_descriptor_t descriptor = make_descriptor(
        3U, rank, SAGR_CCL_V1_OPERATION_ALL_REDUCE,
        SAGR_CCL_V1_DTYPE_FP32, SAGR_CCL_V1_REDUCTION_SUM, 12U, 12U);
    REQUIRE(sagr_ccl_v1_group_state_begin(&state, &descriptor) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
  }
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ACTIVE);
  for (rank = 2U;; --rank) {
    REQUIRE(sagr_ccl_v1_group_state_complete(&state, rank, 1U) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    if (rank == 0U) {
      break;
    }
  }
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_READY &&
          state.next_sequence == 2U && state.active_sequence == 0U);
  REQUIRE(sagr_ccl_v1_group_state_snapshot(
              &state, &snapshot, (uint32_t)sizeof(snapshot)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(snapshot.next_sequence == 2U && snapshot.joined_mask == 7U &&
          snapshot.begun_mask == 0U && snapshot.completed_mask == 0U);
  REQUIRE(sagr_ccl_v1_group_state_close(&state) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_group_state_begin(&state, NULL) ==
          SAGR_CCL_V1_STATUS_CLOSED);
  return 0;
}

static int test_state_machine_world_matrix(void) {
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    sagr_ccl_v1_group_identity_t identity = make_identity(world);
    sagr_ccl_v1_group_state_t state;
    uint32_t rank;
    REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                         &identity) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(join_all(&state, &identity) == 0);
    for (rank = 0U; rank < world; ++rank) {
      sagr_ccl_v1_descriptor_t descriptor = make_descriptor(
          world, rank, SAGR_CCL_V1_OPERATION_BARRIER,
          SAGR_CCL_V1_DTYPE_NONE, SAGR_CCL_V1_REDUCTION_NONE, 0U, 0U);
      REQUIRE(sagr_ccl_v1_group_state_begin(&state, &descriptor) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
    }
    REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ACTIVE);
    for (rank = 0U; rank < world; ++rank) {
      REQUIRE(sagr_ccl_v1_group_state_complete(&state, rank, 1U) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
    }
    REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_READY &&
            state.next_sequence == 2U);
    REQUIRE(sagr_ccl_v1_group_state_close(&state) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
  }
  return 0;
}

typedef enum state_fixture_kind {
  STATE_FIXTURE_JOINING = 0,
  STATE_FIXTURE_READY,
  STATE_FIXTURE_COLLECTING,
  STATE_FIXTURE_ACTIVE,
  STATE_FIXTURE_ABORTED,
  STATE_FIXTURE_CLOSED
} state_fixture_kind_t;

typedef enum state_mutation_kind {
  STATE_MUTATION_JOIN = 0,
  STATE_MUTATION_BEGIN,
  STATE_MUTATION_COMPLETE,
  STATE_MUTATION_ABORT,
  STATE_MUTATION_CLOSE
} state_mutation_kind_t;

typedef enum state_corruption_kind {
  STATE_CORRUPT_RESERVED0 = 0,
  STATE_CORRUPT_RESERVED_BYTES,
  STATE_CORRUPT_JOINED_OUTSIDE_WORLD,
  STATE_CORRUPT_BEGUN_OUTSIDE_WORLD,
  STATE_CORRUPT_COMPLETED_OUTSIDE_WORLD,
  STATE_CORRUPT_COMPLETED_NOT_BEGUN,
  STATE_CORRUPT_INVALID_PHASE,
  STATE_CORRUPT_READY_JOINED_PARTIAL,
  STATE_CORRUPT_READY_ACTIVE_SEQUENCE,
  STATE_CORRUPT_READY_ACTIVE_DESCRIPTOR,
  STATE_CORRUPT_JOINING_JOINED_FULL,
  STATE_CORRUPT_COLLECTING_BEGUN_ZERO,
  STATE_CORRUPT_COLLECTING_BEGUN_FULL,
  STATE_CORRUPT_COLLECTING_COMPLETED,
  STATE_CORRUPT_ACTIVE_BEGUN_PARTIAL,
  STATE_CORRUPT_ACTIVE_COMPLETED_FULL,
  STATE_CORRUPT_ACTIVE_SEQUENCE,
  STATE_CORRUPT_ACTIVE_DESCRIPTOR_SEQUENCE,
  STATE_CORRUPT_ACTIVE_DESCRIPTOR_RANK,
  STATE_CORRUPT_ACTIVE_DESCRIPTOR_RESERVED,
  STATE_CORRUPT_ACTIVE_DESCRIPTOR_IDENTITY,
  STATE_CORRUPT_ACTIVE_ABORT_RANK,
  STATE_CORRUPT_ACTIVE_ABORT_STATUS,
  STATE_CORRUPT_ABORTED_SUCCESS_STATUS,
  STATE_CORRUPT_ABORTED_RANK_OUTSIDE_WORLD,
  STATE_CORRUPT_ABORTED_ACTIVE_SEQUENCE,
  STATE_CORRUPT_CLOSED_JOINED_PARTIAL,
  STATE_CORRUPT_NEXT_SEQUENCE_ZERO,
  STATE_CORRUPT_NEXT_SEQUENCE_RESERVED
} state_corruption_kind_t;

typedef struct state_corruption_case {
  state_fixture_kind_t fixture;
  state_corruption_kind_t corruption;
} state_corruption_case_t;

static uint32_t test_world_mask(uint32_t world_size) {
  return (UINT32_C(1) << world_size) - UINT32_C(1);
}

static int prepare_state_fixture(
    uint32_t world_size, state_fixture_kind_t fixture,
    sagr_ccl_v1_group_identity_t *identity,
    sagr_ccl_v1_group_state_t *state) {
  uint32_t rank;
  *identity = make_identity(world_size);
  REQUIRE(sagr_ccl_v1_group_state_init(state, (uint32_t)sizeof(*state),
                                       identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  if (fixture == STATE_FIXTURE_JOINING) {
    return 0;
  }
  REQUIRE(join_all(state, identity) == 0);
  if (fixture == STATE_FIXTURE_READY) {
    return 0;
  }
  if (fixture == STATE_FIXTURE_COLLECTING ||
      fixture == STATE_FIXTURE_ACTIVE) {
    const uint32_t begin_count =
        fixture == STATE_FIXTURE_COLLECTING ? 1U : world_size;
    for (rank = 0U; rank < begin_count; ++rank) {
      sagr_ccl_v1_descriptor_t descriptor = make_descriptor(
          world_size, rank, SAGR_CCL_V1_OPERATION_BARRIER,
          SAGR_CCL_V1_DTYPE_NONE, SAGR_CCL_V1_REDUCTION_NONE, 0U, 0U);
      descriptor.group = *identity;
      REQUIRE(sagr_ccl_v1_group_state_begin(state, &descriptor) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
    }
    return 0;
  }
  if (fixture == STATE_FIXTURE_ABORTED) {
    REQUIRE(sagr_ccl_v1_group_state_abort(
                state, 0U, state->next_sequence,
                SAGR_CCL_V1_STATUS_PEER_LOST) ==
            SAGR_CCL_V1_STATUS_PEER_LOST);
    return 0;
  }
  REQUIRE(fixture == STATE_FIXTURE_CLOSED);
  REQUIRE(sagr_ccl_v1_group_state_close(state) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  return 0;
}

static sagr_ccl_v1_status_t invoke_state_mutation(
    state_mutation_kind_t mutation,
    sagr_ccl_v1_group_identity_t *identity,
    sagr_ccl_v1_group_state_t *state) {
  sagr_ccl_v1_descriptor_t descriptor;
  switch (mutation) {
    case STATE_MUTATION_JOIN:
      return sagr_ccl_v1_group_state_join(state, identity, 0U);
    case STATE_MUTATION_BEGIN:
      descriptor = make_descriptor(
          identity->world_size, 0U, SAGR_CCL_V1_OPERATION_BARRIER,
          SAGR_CCL_V1_DTYPE_NONE, SAGR_CCL_V1_REDUCTION_NONE, 0U, 0U);
      descriptor.group = *identity;
      return sagr_ccl_v1_group_state_begin(state, &descriptor);
    case STATE_MUTATION_COMPLETE:
      return sagr_ccl_v1_group_state_complete(
          state, 0U, state->active_sequence);
    case STATE_MUTATION_ABORT:
      return sagr_ccl_v1_group_state_abort(
          state, 0U, state->next_sequence,
          SAGR_CCL_V1_STATUS_PEER_LOST);
    case STATE_MUTATION_CLOSE:
      return sagr_ccl_v1_group_state_close(state);
    default:
      return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
}

static int require_canonical_protocol_abort(
    const sagr_ccl_v1_group_state_t *state) {
  sagr_ccl_v1_descriptor_t empty_descriptor;
  sagr_ccl_v1_group_snapshot_t snapshot;
  memset(&empty_descriptor, 0, sizeof(empty_descriptor));
  REQUIRE(state->phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED);
  REQUIRE(state->active_sequence == 0U && state->begun_mask == 0U &&
          state->completed_mask == 0U);
  REQUIRE(state->abort_rank == SAGR_CCL_V1_NO_RANK &&
          state->abort_status == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  REQUIRE(state->reserved0 == 0U &&
          memcmp(state->reserved, empty_descriptor.reserved,
                 sizeof(state->reserved)) == 0);
  REQUIRE(memcmp(&state->active_descriptor, &empty_descriptor,
                 sizeof(empty_descriptor)) == 0);
  REQUIRE((state->joined_mask &
           ~test_world_mask(state->identity.world_size)) == 0U);
  REQUIRE(state->next_sequence != 0U);
  REQUIRE(sagr_ccl_v1_group_state_snapshot(
              state, &snapshot, (uint32_t)sizeof(snapshot)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  return 0;
}

static void corrupt_state(sagr_ccl_v1_group_state_t *state,
                          state_corruption_kind_t corruption) {
  const uint32_t mask = test_world_mask(state->identity.world_size);
  const uint32_t outside = UINT32_C(1) << state->identity.world_size;
  switch (corruption) {
    case STATE_CORRUPT_RESERVED0:
      state->reserved0 = 1U;
      break;
    case STATE_CORRUPT_RESERVED_BYTES:
      state->reserved[sizeof(state->reserved) - 1U] = 1U;
      break;
    case STATE_CORRUPT_JOINED_OUTSIDE_WORLD:
      state->joined_mask |= outside;
      break;
    case STATE_CORRUPT_BEGUN_OUTSIDE_WORLD:
      state->begun_mask |= outside;
      break;
    case STATE_CORRUPT_COMPLETED_OUTSIDE_WORLD:
      state->completed_mask |= outside;
      break;
    case STATE_CORRUPT_COMPLETED_NOT_BEGUN:
      state->completed_mask = 1U;
      break;
    case STATE_CORRUPT_INVALID_PHASE:
      state->phase = UINT32_MAX;
      break;
    case STATE_CORRUPT_READY_JOINED_PARTIAL:
      state->joined_mask &= ~UINT32_C(1);
      break;
    case STATE_CORRUPT_READY_ACTIVE_SEQUENCE:
      state->active_sequence = state->next_sequence;
      break;
    case STATE_CORRUPT_READY_ACTIVE_DESCRIPTOR:
      state->active_descriptor.struct_size =
          (uint32_t)sizeof(state->active_descriptor);
      break;
    case STATE_CORRUPT_JOINING_JOINED_FULL:
      state->joined_mask = mask;
      break;
    case STATE_CORRUPT_COLLECTING_BEGUN_ZERO:
      state->begun_mask = 0U;
      break;
    case STATE_CORRUPT_COLLECTING_BEGUN_FULL:
      state->begun_mask = mask;
      break;
    case STATE_CORRUPT_COLLECTING_COMPLETED:
      state->completed_mask = 1U;
      break;
    case STATE_CORRUPT_ACTIVE_BEGUN_PARTIAL:
      state->begun_mask &= ~UINT32_C(1);
      break;
    case STATE_CORRUPT_ACTIVE_COMPLETED_FULL:
      state->completed_mask = mask;
      break;
    case STATE_CORRUPT_ACTIVE_SEQUENCE:
      ++state->active_sequence;
      break;
    case STATE_CORRUPT_ACTIVE_DESCRIPTOR_SEQUENCE:
      ++state->active_descriptor.sequence;
      break;
    case STATE_CORRUPT_ACTIVE_DESCRIPTOR_RANK:
      state->active_descriptor.rank = 0U;
      break;
    case STATE_CORRUPT_ACTIVE_DESCRIPTOR_RESERVED:
      state->active_descriptor.reserved[0] = 1U;
      break;
    case STATE_CORRUPT_ACTIVE_DESCRIPTOR_IDENTITY:
      ++state->active_descriptor.group.group_generation;
      break;
    case STATE_CORRUPT_ACTIVE_ABORT_RANK:
      state->abort_rank = 0U;
      break;
    case STATE_CORRUPT_ACTIVE_ABORT_STATUS:
      state->abort_status = SAGR_CCL_V1_STATUS_PEER_LOST;
      break;
    case STATE_CORRUPT_ABORTED_SUCCESS_STATUS:
      state->abort_status = SAGR_CCL_V1_STATUS_SUCCESS;
      break;
    case STATE_CORRUPT_ABORTED_RANK_OUTSIDE_WORLD:
      state->abort_rank = state->identity.world_size;
      break;
    case STATE_CORRUPT_ABORTED_ACTIVE_SEQUENCE:
      state->active_sequence = state->next_sequence;
      break;
    case STATE_CORRUPT_CLOSED_JOINED_PARTIAL:
      state->joined_mask &= ~UINT32_C(1);
      break;
    case STATE_CORRUPT_NEXT_SEQUENCE_ZERO:
      state->next_sequence = 0U;
      break;
    case STATE_CORRUPT_NEXT_SEQUENCE_RESERVED:
      state->next_sequence = UINT64_MAX;
      break;
  }
}

static int test_every_mutation_validates_state_first(void) {
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    state_mutation_kind_t mutation;
    for (mutation = STATE_MUTATION_JOIN;
         mutation <= STATE_MUTATION_CLOSE; ++mutation) {
      sagr_ccl_v1_group_identity_t identity;
      sagr_ccl_v1_group_state_t state;
      state_fixture_kind_t fixture = STATE_FIXTURE_READY;
      if (mutation == STATE_MUTATION_JOIN) {
        fixture = STATE_FIXTURE_JOINING;
      } else if (mutation == STATE_MUTATION_COMPLETE) {
        fixture = STATE_FIXTURE_ACTIVE;
      }
      REQUIRE(prepare_state_fixture(world, fixture, &identity, &state) == 0);
      state.reserved[0] = 1U;
      REQUIRE(invoke_state_mutation(mutation, &identity, &state) ==
              SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
      REQUIRE(require_canonical_protocol_abort(&state) == 0);
    }
  }
  return 0;
}

static int test_state_corruption_matrix(void) {
  static const state_corruption_case_t cases[] = {
      {STATE_FIXTURE_READY, STATE_CORRUPT_RESERVED0},
      {STATE_FIXTURE_READY, STATE_CORRUPT_RESERVED_BYTES},
      {STATE_FIXTURE_READY, STATE_CORRUPT_JOINED_OUTSIDE_WORLD},
      {STATE_FIXTURE_READY, STATE_CORRUPT_BEGUN_OUTSIDE_WORLD},
      {STATE_FIXTURE_READY, STATE_CORRUPT_COMPLETED_OUTSIDE_WORLD},
      {STATE_FIXTURE_READY, STATE_CORRUPT_COMPLETED_NOT_BEGUN},
      {STATE_FIXTURE_READY, STATE_CORRUPT_INVALID_PHASE},
      {STATE_FIXTURE_READY, STATE_CORRUPT_READY_JOINED_PARTIAL},
      {STATE_FIXTURE_READY, STATE_CORRUPT_READY_ACTIVE_SEQUENCE},
      {STATE_FIXTURE_READY, STATE_CORRUPT_READY_ACTIVE_DESCRIPTOR},
      {STATE_FIXTURE_JOINING, STATE_CORRUPT_JOINING_JOINED_FULL},
      {STATE_FIXTURE_COLLECTING, STATE_CORRUPT_COLLECTING_BEGUN_ZERO},
      {STATE_FIXTURE_COLLECTING, STATE_CORRUPT_COLLECTING_BEGUN_FULL},
      {STATE_FIXTURE_COLLECTING, STATE_CORRUPT_COLLECTING_COMPLETED},
      {STATE_FIXTURE_ACTIVE, STATE_CORRUPT_ACTIVE_BEGUN_PARTIAL},
      {STATE_FIXTURE_ACTIVE, STATE_CORRUPT_ACTIVE_COMPLETED_FULL},
      {STATE_FIXTURE_ACTIVE, STATE_CORRUPT_ACTIVE_SEQUENCE},
      {STATE_FIXTURE_ACTIVE, STATE_CORRUPT_ACTIVE_DESCRIPTOR_SEQUENCE},
      {STATE_FIXTURE_ACTIVE, STATE_CORRUPT_ACTIVE_DESCRIPTOR_RANK},
      {STATE_FIXTURE_ACTIVE, STATE_CORRUPT_ACTIVE_DESCRIPTOR_RESERVED},
      {STATE_FIXTURE_ACTIVE, STATE_CORRUPT_ACTIVE_DESCRIPTOR_IDENTITY},
      {STATE_FIXTURE_ACTIVE, STATE_CORRUPT_ACTIVE_ABORT_RANK},
      {STATE_FIXTURE_ACTIVE, STATE_CORRUPT_ACTIVE_ABORT_STATUS},
      {STATE_FIXTURE_ABORTED, STATE_CORRUPT_ABORTED_SUCCESS_STATUS},
      {STATE_FIXTURE_ABORTED, STATE_CORRUPT_ABORTED_RANK_OUTSIDE_WORLD},
      {STATE_FIXTURE_ABORTED, STATE_CORRUPT_ABORTED_ACTIVE_SEQUENCE},
      {STATE_FIXTURE_CLOSED, STATE_CORRUPT_CLOSED_JOINED_PARTIAL},
      {STATE_FIXTURE_READY, STATE_CORRUPT_NEXT_SEQUENCE_ZERO},
      {STATE_FIXTURE_READY, STATE_CORRUPT_NEXT_SEQUENCE_RESERVED},
  };
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    size_t index;
    for (index = 0U; index < sizeof(cases) / sizeof(cases[0]); ++index) {
      sagr_ccl_v1_group_identity_t identity;
      sagr_ccl_v1_group_snapshot_t snapshot;
      sagr_ccl_v1_group_state_t state;
      REQUIRE(prepare_state_fixture(world, cases[index].fixture, &identity,
                                    &state) == 0);
      corrupt_state(&state, cases[index].corruption);
      REQUIRE(sagr_ccl_v1_group_state_snapshot(
                  &state, &snapshot, (uint32_t)sizeof(snapshot)) ==
              SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
      REQUIRE(sagr_ccl_v1_group_state_close(&state) ==
              SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
      REQUIRE(require_canonical_protocol_abort(&state) == 0);
    }
  }
  return 0;
}

static int test_corrupted_state_identity_fails_closed(void) {
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    sagr_ccl_v1_group_identity_t identity;
    sagr_ccl_v1_group_state_t state;
    REQUIRE(prepare_state_fixture(world, STATE_FIXTURE_ACTIVE, &identity,
                                  &state) == 0);
    state.identity.group_generation = 0U;
    state.reserved0 = 1U;
    REQUIRE(sagr_ccl_v1_group_state_complete(
                &state, 0U, state.active_sequence) ==
            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
    REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED);
    REQUIRE(state.active_sequence == 0U && state.begun_mask == 0U &&
            state.completed_mask == 0U);
    REQUIRE(state.abort_rank == SAGR_CCL_V1_NO_RANK &&
            state.abort_status == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
    REQUIRE(state.reserved0 == 0U);
    REQUIRE(sagr_ccl_v1_group_state_close(&state) ==
            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  }
  return 0;
}

static int test_sequence_exhaustion_fails_closed(void) {
  sagr_ccl_v1_group_identity_t identity;
  sagr_ccl_v1_group_state_t state;
  uint32_t rank;
  REQUIRE(prepare_state_fixture(3U, STATE_FIXTURE_ACTIVE, &identity, &state) ==
          0);
  state.next_sequence = UINT64_MAX - UINT64_C(1);
  state.active_sequence = state.next_sequence;
  state.active_descriptor.sequence = state.next_sequence;
  for (rank = 0U; rank + 1U < identity.world_size; ++rank) {
    REQUIRE(sagr_ccl_v1_group_state_complete(
                &state, rank, state.active_sequence) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
  }
  REQUIRE(sagr_ccl_v1_group_state_complete(
              &state, rank, state.active_sequence) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_rank == rank &&
          state.abort_status == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  REQUIRE(state.next_sequence == UINT64_MAX - UINT64_C(1));
  return 0;
}

static int test_join_fail_closed(void) {
  sagr_ccl_v1_group_identity_t identity = make_identity(3U);
  sagr_ccl_v1_group_identity_t stale = identity;
  sagr_ccl_v1_group_state_t state;
  stale.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_group_state_join(&state, &stale, 0U) ==
          SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_rank == 0U &&
          state.abort_status == SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  REQUIRE(sagr_ccl_v1_group_state_join(&state, &identity, 0U) ==
          SAGR_CCL_V1_STATUS_ABORTED);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_group_state_join(&state, &identity, 0U) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_group_state_join(&state, &identity, 0U) ==
          SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_rank == 0U &&
          state.abort_status == SAGR_CCL_V1_STATUS_OUT_OF_ORDER);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_group_state_join(&state, &identity, 3U) ==
          SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_rank == 3U &&
          state.abort_status == SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(join_all(&state, &identity) == 0);
  REQUIRE(sagr_ccl_v1_group_state_join(&state, &identity, 0U) ==
          SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_rank == 0U &&
          state.abort_status == SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  return 0;
}

static int test_state_machine_fail_closed(void) {
  sagr_ccl_v1_group_identity_t identity = make_identity(3U);
  sagr_ccl_v1_group_state_t state;
  sagr_ccl_v1_descriptor_t rank0;
  sagr_ccl_v1_descriptor_t rank1;
  sagr_ccl_v1_group_snapshot_t snapshot;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(join_all(&state, &identity) == 0);
  rank0 = make_descriptor(3U, 0U, SAGR_CCL_V1_OPERATION_ALL_REDUCE,
                          SAGR_CCL_V1_DTYPE_BF16,
                          SAGR_CCL_V1_REDUCTION_SUM, 12U, 12U);
  rank1 = rank0;
  rank1.rank = 1U;
  rank1.input_count = 13U;
  rank1.output_count = 13U;
  REQUIRE(sagr_ccl_v1_group_state_begin(&state, &rank0) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_group_state_begin(&state, &rank1) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_rank == 1U &&
          state.abort_status == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  REQUIRE(sagr_ccl_v1_group_state_begin(&state, &rank0) ==
          SAGR_CCL_V1_STATUS_ABORTED);
  REQUIRE(sagr_ccl_v1_group_state_snapshot(
              &state, &snapshot, (uint32_t)sizeof(snapshot)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(snapshot.abort_status == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  REQUIRE(sagr_ccl_v1_group_state_close(&state) ==
          SAGR_CCL_V1_STATUS_SUCCESS);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(join_all(&state, &identity) == 0);
  rank0 = make_descriptor(3U, 0U, SAGR_CCL_V1_OPERATION_ALL_GATHER,
                          SAGR_CCL_V1_DTYPE_UINT8,
                          SAGR_CCL_V1_REDUCTION_NONE, 4U, 12U);
  rank0.group = identity;
  rank0.group.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_begin(&state, &rank0) ==
          SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_status == SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(join_all(&state, &identity) == 0);
  rank0 = make_descriptor(3U, 0U, SAGR_CCL_V1_OPERATION_ALL_REDUCE,
                          SAGR_CCL_V1_DTYPE_INT32,
                          SAGR_CCL_V1_REDUCTION_SUM, 4U, 4U);
  rank0.group = identity;
  REQUIRE(sagr_ccl_v1_group_state_begin(&state, &rank0) ==
          SAGR_CCL_V1_STATUS_NOT_SUPPORTED);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_status == SAGR_CCL_V1_STATUS_NOT_SUPPORTED);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(join_all(&state, &identity) == 0);
  rank0 = make_descriptor(3U, 0U, SAGR_CCL_V1_OPERATION_BARRIER,
                          SAGR_CCL_V1_DTYPE_NONE,
                          SAGR_CCL_V1_REDUCTION_NONE, 0U, 0U);
  rank0.group = identity;
  rank0.sequence = 2U;
  REQUIRE(sagr_ccl_v1_group_state_begin(&state, &rank0) ==
          SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(join_all(&state, &identity) == 0);
  REQUIRE(sagr_ccl_v1_group_state_abort(
              &state, 2U, 1U, SAGR_CCL_V1_STATUS_PEER_LOST) ==
          SAGR_CCL_V1_STATUS_PEER_LOST);
  REQUIRE(state.abort_status == SAGR_CCL_V1_STATUS_PEER_LOST);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  rank0 = make_descriptor(3U, 0U, SAGR_CCL_V1_OPERATION_BARRIER,
                          SAGR_CCL_V1_DTYPE_NONE,
                          SAGR_CCL_V1_REDUCTION_NONE, 0U, 0U);
  rank0.group = identity;
  REQUIRE(sagr_ccl_v1_group_state_begin(&state, &rank0) ==
          SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_status == SAGR_CCL_V1_STATUS_OUT_OF_ORDER);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_group_state_abort(
              &state, 3U, 1U, SAGR_CCL_V1_STATUS_PEER_LOST) ==
          SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_status == SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH);

  identity.epoch += 1U;
  identity.group_generation += 1U;
  REQUIRE(sagr_ccl_v1_group_state_init(&state, (uint32_t)sizeof(state),
                                       &identity) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_v1_group_state_abort(
              &state, 0U, 1U, SAGR_CCL_V1_STATUS_INVALID_ARGUMENT) ==
          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  REQUIRE(state.phase == SAGR_CCL_V1_GROUP_PHASE_ABORTED &&
          state.abort_status == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR);
  return 0;
}

int main(void) {
  REQUIRE(test_oversized_init_does_not_touch_trailing_memory() == 0);
  REQUIRE(test_identity_and_descriptor_legality() == 0);
  REQUIRE(test_codec_roundtrip_and_corruption() == 0);
  REQUIRE(test_planner_matrix() == 0);
  REQUIRE(test_state_machine_success() == 0);
  REQUIRE(test_state_machine_world_matrix() == 0);
  REQUIRE(test_every_mutation_validates_state_first() == 0);
  REQUIRE(test_state_corruption_matrix() == 0);
  REQUIRE(test_corrupted_state_identity_fails_closed() == 0);
  REQUIRE(test_sequence_exhaustion_fails_closed() == 0);
  REQUIRE(test_join_fail_closed() == 0);
  REQUIRE(test_state_machine_fail_closed() == 0);
  puts("ccl_v1: all standalone API/codec/planner/state-machine tests passed");
  return 0;
}
