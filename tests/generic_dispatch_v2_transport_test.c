/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "transport_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int expect(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "generic dispatch v2 transport: %s\n", message);
    return 1;
  }
  return 0;
}

static int load_vector(const char *name,
                       uint8_t output[SAGR_WIRE_GENERIC_PAYLOAD_BYTES]) {
  char path[1024];
  FILE *stream;
  size_t bytes;
  int written = snprintf(path, sizeof(path), "%s/%s",
                         AMDGPU_SIM_BRIDGE_VECTOR_DIR, name);
  if (written <= 0 || (size_t)written >= sizeof(path)) {
    return 0;
  }
  stream = fopen(path, "rb");
  if (stream == NULL) {
    return 0;
  }
  bytes = fread(output, 1U, SAGR_WIRE_GENERIC_PAYLOAD_BYTES, stream);
  return fgetc(stream) == EOF && fclose(stream) == 0 &&
         bytes == SAGR_WIRE_GENERIC_PAYLOAD_BYTES;
}

static int expect_payload_vector(const uint8_t *frame, const char *name) {
  uint8_t expected[SAGR_WIRE_GENERIC_PAYLOAD_BYTES];
  size_t mismatch = SAGR_WIRE_GENERIC_PAYLOAD_BYTES;
  if (!load_vector(name, expected)) {
    return expect(0, "shared response vector loads");
  }
  for (size_t index = 0; index < SAGR_WIRE_GENERIC_PAYLOAD_BYTES; ++index) {
    if (frame[SAGR_WIRE_HEADER_BYTES + index] != expected[index]) {
      mismatch = index;
      break;
    }
  }
  if (mismatch == SAGR_WIRE_GENERIC_PAYLOAD_BYTES) {
    return 0;
  }
  fprintf(stderr,
          "generic dispatch v2 transport: %s differs at payload offset %zu "
          "(actual=0x%02x expected=0x%02x)\n",
          name, mismatch, frame[SAGR_WIRE_HEADER_BYTES + mismatch],
          expected[mismatch]);
  return 1;
}

static void fill_info(sagr_instance_info_t *info) {
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->selected_version_major = 1U;
  info->maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  info->daemon_uuid[0] = 0x11U;
  info->job_uuid[0] = 0x22U;
  info->connection_id = UINT64_C(0x1234);
  info->epoch = UINT64_C(0x5678);
  info->negotiated_capabilities[0] =
      SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK |
      SAGR_CAPABILITY_MEMORY_MASK | SAGR_CAPABILITY_SIGNAL_MASK |
      SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK |
      SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
}

static void fill_request(sagr_wire_generic_request_t *request, uint16_t opcode) {
  size_t index;
  memset(request, 0, sizeof(*request));
  request->major = SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR;
  request->minor = SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR;
  request->opcode = opcode;
  request->object_id = UINT64_C(7);
  request->object_generation = UINT64_C(9);
  request->kernel_index = 0U;
  for (index = 0; index < sizeof(request->image_sha256); ++index) {
    request->image_sha256[index] = (uint8_t)(index + 1U);
  }
  (void)snprintf(request->kernel_name, sizeof(request->kernel_name), "%s",
                 "bridge_generic_kernel");
}

static int round_trip_request(const sagr_instance_info_t *info,
                              const sagr_wire_generic_request_t *request,
                              sagr_wire_generic_request_t *decoded,
                              uint64_t expected_id,
                              const char *vector_name) {
  uint8_t frame[SAGR_WIRE_GENERIC_FRAME_BYTES];
  uint8_t expected_payload[SAGR_WIRE_GENERIC_PAYLOAD_BYTES];
  size_t frame_size = 0U;
  uint64_t request_id = 0U;
  const char *reason = NULL;
  int failures = 0;
  size_t mismatch = SAGR_WIRE_GENERIC_PAYLOAD_BYTES;
  failures += expect(sagr_protocol_encode_generic_dispatch_request(
                         info, expected_id, request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS &&
                         frame_size == SAGR_WIRE_GENERIC_FRAME_BYTES,
                     "request uses one fixed 4096-byte frame");
  failures += expect(load_vector(vector_name, expected_payload),
                     "shared runtime-gem5 golden vector loads");
  for (size_t index = 0; index < SAGR_WIRE_GENERIC_PAYLOAD_BYTES; ++index) {
    if (frame[SAGR_WIRE_HEADER_BYTES + index] != expected_payload[index]) {
      mismatch = index;
      break;
    }
  }
  if (mismatch != SAGR_WIRE_GENERIC_PAYLOAD_BYTES) {
    fprintf(stderr,
            "generic dispatch v2 transport: %s differs at payload offset %zu "
            "(actual=0x%02x expected=0x%02x)\n",
            vector_name, mismatch, frame[SAGR_WIRE_HEADER_BYTES + mismatch],
            expected_payload[mismatch]);
    ++failures;
  }
  failures += expect(sagr_protocol_decode_generic_dispatch_request(
                         frame, frame_size, info, decoded, &request_id,
                         &reason) == SAGR_STATUS_SUCCESS &&
                         request_id == expected_id &&
                         decoded->opcode == request->opcode,
                     "request round-trips with header correlation");
  return failures;
}

static void fill_map_response(sagr_wire_generic_response_t *response) {
  size_t index;
  memset(response, 0, sizeof(*response));
  response->major = SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR;
  response->minor = SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR;
  response->status = SAGR_WIRE_STATUS_OK;
  response->opcode = SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT;
  response->object_id = UINT64_C(7);
  response->object_generation = UINT64_C(9);
  response->mapping_id = UINT64_C(11);
  response->mapping_generation = UINT64_C(13);
  response->mapped_base_va = UINT64_C(0x100000000000);
  response->mapped_end_va = UINT64_C(0x100000003000);
  response->descriptor_va = UINT64_C(0x1000000005c0);
  response->code_va = UINT64_C(0x100000001600);
  response->entry_va = UINT64_C(0x100000001600);
  response->mapped_bytes = UINT64_C(0x3000);
  response->kernel_index = 0U;
  response->segment_count = 3U;
  response->descriptor_preload_dwords = 12U;
  for (index = 0; index < sizeof(response->image_sha256); ++index) {
    response->image_sha256[index] = (uint8_t)(index + 1U);
  }
}

int main(void) {
  sagr_instance_info_t info;
  sagr_wire_generic_request_t request;
  sagr_wire_generic_request_t decoded_request;
  sagr_wire_generic_response_t response;
  sagr_wire_generic_response_t decoded_response;
  uint8_t frame[SAGR_WIRE_GENERIC_FRAME_BYTES];
  uint8_t mutated[SAGR_WIRE_GENERIC_FRAME_BYTES];
  size_t frame_size = 0U;
  uint64_t request_id = 0U;
  int32_t wire_status = -1;
  const char *reason = NULL;
  int failures = 0;

  fill_info(&info);

  fill_request(&request, SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT);
  request.body.map.gfx_target = 950U;
  request.body.map.kernarg_segment_size = 48U;
  request.body.map.kernarg_segment_align = 8U;
  request.body.map.descriptor_preload_dwords = 12U;
  request.body.map.page_size = 4096U;
  failures += round_trip_request(
      &info, &request, &decoded_request, 41U,
      "request-map-object.bin");
  failures += expect(decoded_request.body.map.descriptor_preload_dwords == 12U,
                     "Triton 12-DWORD preload round-trips exactly");
  request.body.map.descriptor_preload_dwords =
      SAGR_WIRE_GENERIC_MAX_PRELOAD_DWORDS + 1U;
  failures += expect(sagr_protocol_encode_generic_dispatch_request(
                         &info, 42U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                     "preload above 64 DWORD is rejected");
  request.body.map.descriptor_preload_dwords = 12U;
  failures += round_trip_request(
      &info, &request, &decoded_request, 42U,
      "request-map-object.bin");

  /* Re-encode before mutating so this check is independent of the helper's
   * local frame lifetime. */
  failures += expect(sagr_protocol_encode_generic_dispatch_request(
                         &info, 43U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS,
                     "MAP re-encodes before mutation");
  failures += expect(frame[14] == 0U && frame[15] ==
                         SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_REQUEST &&
                         frame[SAGR_WIRE_HEADER_BYTES] == 0U &&
                         frame[SAGR_WIRE_HEADER_BYTES + 1U] == 2U,
                     "v2 request uses the v1 envelope and big-endian payload");
  memcpy(mutated, frame, sizeof(mutated));
  mutated[SAGR_WIRE_HEADER_BYTES + 4U] ^= 1U;
  failures += expect(sagr_protocol_decode_generic_dispatch_request(
                         mutated, sizeof(mutated), &info, &decoded_request,
                         &request_id, &reason) == SAGR_STATUS_CHECKSUM_ERROR,
                     "request CRC protects the payload");
  memcpy(mutated, frame, sizeof(mutated));
  mutated[SAGR_WIRE_HEADER_BYTES + 4015U] = 0xa5U;
  sagr_protocol_recompute_frame_crc(mutated, sizeof(mutated));
  failures += expect(sagr_protocol_decode_generic_dispatch_request(
                         mutated, sizeof(mutated), &info, &decoded_request,
                         &request_id, &reason) == SAGR_STATUS_PROTOCOL_ERROR,
                     "nonzero request tail is rejected");

  fill_request(&request, SAGR_WIRE_GENERIC_OPCODE_ALLOC_KERNARG);
  request.mapping_id = UINT64_C(11);
  request.mapping_generation = UINT64_C(13);
  memset(request.image_sha256, 0, sizeof(request.image_sha256));
  memset(request.kernel_name, 0, sizeof(request.kernel_name));
  request.body.alloc_kernarg.size_bytes = 48U;
  request.body.alloc_kernarg.alignment_bytes = 8U;
  failures += round_trip_request(
      &info, &request, &decoded_request, 43U,
      "request-alloc-kernarg.bin");
  failures += expect(decoded_request.body.alloc_kernarg.size_bytes == 48U,
                     "ALLOC preserves kernarg size");

  fill_request(&request, SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL);
  request.mapping_id = UINT64_C(11);
  request.mapping_generation = UINT64_C(13);
  request.queue_id = UINT64_C(17);
  request.queue_generation = UINT64_C(19);
  request.queue_sequence = UINT64_C(23);
  request.body.submit.kernarg_allocation_id = UINT64_C(29);
  request.body.submit.kernarg_generation = UINT64_C(31);
  request.body.submit.kernarg_size = 48U;
  request.body.submit.signal_id = UINT64_C(37);
  request.body.submit.signal_generation = UINT64_C(41);
  request.body.submit.expected_signal_value_bits = UINT64_C(1);
  request.body.submit.grid_x = 24832U;
  request.body.submit.grid_y = 1U;
  request.body.submit.grid_z = 1U;
  request.body.submit.workgroup_x = 256U;
  request.body.submit.workgroup_y = 1U;
  request.body.submit.workgroup_z = 1U;
  request.body.submit.num_warps = 4U;
  request.body.submit.num_ctas = 1U;
  request.body.submit.wavefront_size = 64U;
  failures += round_trip_request(
      &info, &request, &decoded_request, 47U,
      "request-submit-aql.bin");
  failures += expect(decoded_request.body.submit.grid_x == 24832U &&
                         decoded_request.body.submit.workgroup_x == 256U,
                     "SUBMIT preserves work-item and workgroup geometry");
  request.body.submit.expected_signal_value_bits = 0U;
  failures += expect(sagr_protocol_encode_generic_dispatch_request(
                         &info, 48U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                     "SUBMIT rejects an unexpected initial signal value");
  request.body.submit.expected_signal_value_bits = 1U;
  request.body.submit.grid_x = 128U;
  failures += expect(sagr_protocol_encode_generic_dispatch_request(
                         &info, 49U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                     "SUBMIT rejects a grid smaller than its workgroup");
  request.body.submit.grid_x = 24832U;

  fill_request(&request, SAGR_WIRE_GENERIC_OPCODE_UNMAP_OBJECT);
  request.mapping_id = UINT64_C(11);
  request.mapping_generation = UINT64_C(13);
  memset(request.image_sha256, 0, sizeof(request.image_sha256));
  memset(request.kernel_name, 0, sizeof(request.kernel_name));
  failures += round_trip_request(
      &info, &request, &decoded_request, 53U,
      "request-unmap-object.bin");

  fill_map_response(&response);
  failures += expect(sagr_protocol_encode_generic_dispatch_response(
                         &info, 61U, SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK,
                         &response, frame, sizeof(frame), &frame_size) ==
                         SAGR_STATUS_SUCCESS,
                     "MAP ACK encodes daemon-issued GPU VAs");
  failures += expect_payload_vector(frame, "response-map-object-success.bin");
  failures += expect(sagr_protocol_decode_generic_dispatch_response(
                         frame, frame_size, &info, 61U,
                         SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK,
                         &decoded_response, &wire_status, &reason) ==
                         SAGR_STATUS_SUCCESS &&
                         decoded_response.descriptor_preload_dwords == 12U &&
                         decoded_response.request_id == 61U,
                     "MAP ACK round-trips mapping identity and preload");

  memset(&response, 0, sizeof(response));
  response.major = SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR;
  response.minor = SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = SAGR_WIRE_GENERIC_OPCODE_ALLOC_KERNARG;
  response.object_id = 7U;
  response.object_generation = 9U;
  response.mapping_id = 11U;
  response.mapping_generation = 13U;
  response.kernarg_allocation_id = 29U;
  response.kernarg_generation = 31U;
  response.kernarg_va = UINT64_C(0x100000004000);
  response.kernarg_size = 48U;
  response.kernarg_alignment = 8U;
  for (size_t index = 0; index < sizeof(response.image_sha256); ++index) {
    response.image_sha256[index] = (uint8_t)(index + 1U);
  }
  failures += expect(sagr_protocol_encode_generic_dispatch_response(
                         &info, 67U,
                         SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK,
                         &response, frame, sizeof(frame), &frame_size) ==
                         SAGR_STATUS_SUCCESS,
                     "ALLOC ACK encodes daemon allocation identity");
  failures += expect_payload_vector(frame,
                                    "response-alloc-kernarg-success.bin");

  memset(&response, 0, sizeof(response));
  response.major = SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR;
  response.minor = SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL;
  response.object_id = 7U;
  response.object_generation = 9U;
  response.mapping_id = 11U;
  response.mapping_generation = 13U;
  response.queue_id = 17U;
  response.queue_generation = 19U;
  response.queue_sequence = 23U;
  response.kernarg_allocation_id = 29U;
  response.kernarg_generation = 31U;
  response.kernarg_va = UINT64_C(0x100000004000);
  response.kernarg_size = 48U;
  response.kernarg_alignment = 8U;
  response.signal_id = 37U;
  response.signal_generation = 41U;
  response.signal_value_bits = 1U;
  response.ticket_id = 59U;
  response.trace_id = 67U;
  response.packet_va = UINT64_C(0x100000005000);
  response.packet_crc32c = UINT32_C(0x12345678);
  response.admission_tick = 100U;
  response.start_tick = 110U;
  response.end_tick = 120U;
  response.retire_tick = 130U;
  for (size_t index = 0; index < sizeof(response.image_sha256); ++index) {
    response.image_sha256[index] = (uint8_t)(index + 1U);
  }
  failures += expect(sagr_protocol_encode_generic_dispatch_response(
                         &info, 71U,
                         SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_COMPLETION,
                         &response, frame, sizeof(frame), &frame_size) ==
                         SAGR_STATUS_SUCCESS &&
                         sagr_protocol_decode_generic_dispatch_response(
                             frame, frame_size, &info, 71U,
                             SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_COMPLETION,
                             &decoded_response, &wire_status, &reason) ==
                             SAGR_STATUS_SUCCESS &&
                         decoded_response.retire_tick == 130U,
                     "SUBMIT completion round-trips without execution claim");
  failures += expect_payload_vector(frame, "response-submit-aql-success.bin");

  memset(&response, 0, sizeof(response));
  response.major = SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR;
  response.minor = SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = SAGR_WIRE_GENERIC_OPCODE_UNMAP_OBJECT;
  response.object_id = 7U;
  response.object_generation = 9U;
  response.mapping_id = 11U;
  response.mapping_generation = 13U;
  failures += expect(sagr_protocol_encode_generic_dispatch_response(
                         &info, 72U,
                         SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK,
                         &response, frame, sizeof(frame), &frame_size) ==
                         SAGR_STATUS_SUCCESS,
                     "UNMAP ACK encodes released mapping identity");
  failures += expect_payload_vector(frame, "response-unmap-object-success.bin");

  memset(&response, 0, sizeof(response));
  response.major = SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR;
  response.minor = SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_UNSUPPORTED_VERSION;
  response.opcode = SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT;
  response.error_code = SAGR_WIRE_STATUS_UNSUPPORTED_VERSION;
  failures += expect(sagr_protocol_encode_generic_dispatch_response(
                         &info, 73U, SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK,
                         &response, frame, sizeof(frame), &frame_size) ==
                         SAGR_STATUS_SUCCESS &&
                         sagr_protocol_decode_generic_dispatch_response(
                             frame, frame_size, &info, 73U,
                             SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK,
                             &decoded_response, &wire_status, &reason) ==
                             SAGR_STATUS_VERSION_MISMATCH && wire_status ==
                             SAGR_WIRE_STATUS_UNSUPPORTED_VERSION,
                     "failed ACK is canonical and maps its wire status");
  failures += expect_payload_vector(
      frame, "response-map-object-unsupported-version.bin");

  request.mapping_id = UINT64_C(99);
  request.opcode = SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT;
  request.body.map.gfx_target = 950U;
  request.body.map.kernarg_segment_size = 48U;
  request.body.map.kernarg_segment_align = 8U;
  request.body.map.page_size = 4096U;
  failures += expect(sagr_protocol_encode_generic_dispatch_request(
                         &info, 79U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                     "MAP rejects a client-supplied mapping identity");

  info.negotiated_capabilities[0] &= ~SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
  failures += expect(sagr_protocol_encode_generic_dispatch_request(
                         &info, 83U, &decoded_request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_CAPABILITY_MISMATCH,
                     "v2 codec requires the opt-in capability");

  return failures == 0 ? 0 : 1;
}
