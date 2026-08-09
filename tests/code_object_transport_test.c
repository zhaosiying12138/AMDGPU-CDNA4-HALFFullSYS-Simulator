/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "sha256_internal.h"
#include "transport_internal.h"

#include <self_amdgpu_runtime/code_object.h>

#include <stdio.h>
#include <string.h>

static int
expect(int condition, const char *message)
{
  if (!condition) {
    fprintf(stderr, "code-object transport test: %s\n", message);
    return 1;
  }
  return 0;
}

static void
fill_info(sagr_instance_info_t *info)
{
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->selected_version_major = 1U;
  info->selected_version_minor = 0U;
  info->maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  info->daemon_uuid[0] = 1U;
  info->job_uuid[0] = 2U;
  info->connection_id = UINT64_C(0x1001);
  info->epoch = UINT64_C(0x2002);
  info->negotiated_capabilities[0] =
      SAGR_CAPABILITY_TOPOLOGY_MASK |
      SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK;
}

static void
fill_begin(sagr_wire_code_object_request_t *request, const uint8_t digest[32])
{
  sagr_wire_code_object_begin_t *begin = &request->body.begin;
  memset(request, 0, sizeof(*request));
  request->major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
  request->minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
  request->opcode = SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN;
  begin->image_size = UINT64_C(5672);
  begin->chunk_data_bytes = SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES;
  begin->chunk_count = 2U;
  begin->segment_count = 3U;
  begin->kernel_index = 0U;
  memcpy(begin->image_sha256, digest, 32U);
  begin->elf_machine = SAGR_CODE_OBJECT_ELF_MACHINE_AMDGPU;
  begin->elf_type = SAGR_CODE_OBJECT_ELF_TYPE_DYN;
  begin->elf_osabi = SAGR_CODE_OBJECT_ELF_OSABI_AMDGPU_HSA;
  begin->elf_abi_version = SAGR_CODE_OBJECT_ELF_ABI_VERSION;
  begin->elf_flags = UINT32_C(0x54f);
  begin->gfx_target = SAGR_CODE_OBJECT_TARGET_GFX950;
  begin->code_object_version = 6U;
  begin->metadata_major = 1U;
  begin->metadata_minor = 2U;
  begin->kernarg_segment_size = 288U;
  begin->kernarg_segment_align = 8U;
  begin->max_flat_workgroup_size = 64U;
  begin->wavefront_size = 64U;
  begin->descriptor_size = SAGR_WIRE_CODE_OBJECT_DESCRIPTOR_BYTES;
  begin->descriptor_kernel_code_entry_byte_offset = INT64_C(0x1140);
  begin->code_address = UINT64_C(0x1a00);
  begin->code_file_offset = UINT64_C(0xa00);
  begin->code_size = UINT64_C(0x580);
  begin->descriptor_address = UINT64_C(0x8c0);
  begin->descriptor_file_offset = UINT64_C(0x8c0);
  (void)snprintf(begin->kernel_name, sizeof(begin->kernel_name), "%s", "vecadd");
  (void)snprintf(begin->symbol, sizeof(begin->symbol), "%s", "vecadd.kd");
  begin->segments[0].type = 1U;
  begin->segments[0].flags = 4U;
  begin->segments[0].file_offset = 0U;
  begin->segments[0].virtual_address = 0U;
  begin->segments[0].file_size = UINT64_C(0x984);
  begin->segments[0].memory_size = UINT64_C(0x984);
  begin->segments[0].alignment = UINT64_C(0x1000);
  begin->segments[1].type = 1U;
  begin->segments[1].flags = 5U;
  begin->segments[1].file_offset = UINT64_C(0xa00);
  begin->segments[1].virtual_address = UINT64_C(0x1a00);
  begin->segments[1].file_size = UINT64_C(0x580);
  begin->segments[1].memory_size = UINT64_C(0x580);
  begin->segments[1].alignment = UINT64_C(0x1000);
  begin->segments[2].type = 1U;
  begin->segments[2].flags = 6U;
  begin->segments[2].file_offset = UINT64_C(0xf80);
  begin->segments[2].virtual_address = UINT64_C(0x2f80);
  begin->segments[2].file_size = UINT64_C(0x70);
  begin->segments[2].memory_size = UINT64_C(0x80);
  begin->segments[2].alignment = UINT64_C(0x1000);
}

int
main(void)
{
  static const uint8_t abc_digest[32] = {
      0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
      0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
      0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
  sagr_instance_info_t info;
  sagr_wire_code_object_request_t request;
  sagr_wire_code_object_request_t invalid_begin;
  sagr_wire_code_object_request_t zero_file_request;
  sagr_wire_code_object_request_t decoded_request;
  sagr_wire_code_object_response_t response;
  sagr_wire_code_object_response_t decoded_response;
  uint8_t frame[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  uint8_t mutated[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  uint8_t digest[32];
  uint64_t request_id = 0U;
  size_t frame_size = 0U;
  int failures = 0;
  const char *reason = NULL;
  int32_t wire_status = -1;

  fill_info(&info);
  sagr_sha256("abc", 3U, digest);
  failures += expect(memcmp(digest, abc_digest, sizeof(digest)) == 0,
                    "SHA-256 known vector");
  fill_begin(&request, digest);
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 7U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS &&
                         frame_size == SAGR_WIRE_CODE_OBJECT_FRAME_BYTES,
                    "BEGIN encodes to one fixed record");
  failures += expect(sagr_protocol_decode_code_object_request(
                         frame, frame_size, &info, &decoded_request, &request_id,
                         &reason) == SAGR_STATUS_SUCCESS && request_id == 7U &&
                         decoded_request.body.begin.code_address == 0x1a00U &&
                         decoded_request.body.begin.descriptor_address == 0x8c0U &&
                         decoded_request.body.begin
                                 .descriptor_kernel_code_entry_byte_offset ==
                             0x1140,
                    "BEGIN offsets preserve descriptor-to-code relation");
  zero_file_request = request;
  zero_file_request.body.begin.segments[2].file_size = 0U;
  zero_file_request.body.begin.segments[2].memory_size = 0U;
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 70U, &zero_file_request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS &&
                         sagr_protocol_decode_code_object_request(
                             frame, frame_size, &info, &decoded_request,
                             &request_id, &reason) == SAGR_STATUS_SUCCESS,
                    "BEGIN accepts a zero-file-size segment");
  invalid_begin = request;
  invalid_begin.body.begin.max_flat_workgroup_size = 0U;
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 71U, &invalid_begin, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "BEGIN rejects zero maximum workgroup size");
  invalid_begin = request;
  invalid_begin.body.begin.descriptor_address = UINT64_C(0x8c1);
  invalid_begin.body.begin.descriptor_kernel_code_entry_byte_offset =
      INT64_C(0x113f);
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 72U, &invalid_begin, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "BEGIN rejects an unaligned descriptor address");
  invalid_begin = request;
  invalid_begin.body.begin.segments[2].file_offset = UINT64_C(0x970);
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 73U, &invalid_begin, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "BEGIN rejects overlapping file ranges");
  invalid_begin = request;
  invalid_begin.body.begin.segments[1].flags = 4U;
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 74U, &invalid_begin, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "BEGIN requires code inside an executable segment");
  invalid_begin = request;
  invalid_begin.body.begin.segments[0].flags = 5U;
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 75U, &invalid_begin, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "BEGIN requires descriptor inside a read-only segment");
  invalid_begin = request;
  invalid_begin.body.begin.kernel_name[0] = (char)0x80;
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 76U, &invalid_begin, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "BEGIN rejects non-ASCII kernel names");

  memset(&request, 0, sizeof(request));
  request.major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
  request.opcode = SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK;
  request.minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
  request.object_id = UINT64_C(9);
  request.generation = UINT64_C(10);
  request.image_offset = UINT64_C(3968);
  request.byte_count = 3U;
  request.chunk_index = 1U;
  request.body.chunk[0] = 1U;
  request.body.chunk[1] = 2U;
  request.body.chunk[2] = 3U;
  request.chunk_crc32c = sagr_crc32c(request.body.chunk, 3U);
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 8U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS,
                    "CHUNK encodes with CRC");
  failures += expect(sagr_protocol_decode_code_object_request(
                         frame, frame_size, &info, &decoded_request, &request_id,
                         &reason) == SAGR_STATUS_SUCCESS &&
                         decoded_request.body.chunk[2] == 3U &&
                         decoded_request.body.chunk[3] == 0U,
                    "CHUNK enforces zero padding");
  memcpy(mutated, frame, sizeof(mutated));
  mutated[SAGR_WIRE_HEADER_BYTES + 48U + 3U] = 0xa5U;
  sagr_protocol_recompute_frame_crc(mutated, sizeof(mutated));
  failures += expect(sagr_protocol_decode_code_object_request(
                         mutated, sizeof(mutated), &info, &decoded_request,
                         &request_id, &reason) != SAGR_STATUS_SUCCESS,
                    "nonzero CHUNK tail is rejected");

  memset(&request, 0, sizeof(request));
  request.major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
  request.minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT;
  request.object_id = UINT64_C(9);
  request.generation = UINT64_C(10);
  request.byte_count = 5672U;
  request.chunk_index = 2U;
  memcpy(request.body.commit_sha256, digest, sizeof(digest));
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 9U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS &&
                         sagr_protocol_decode_code_object_request(
                             frame, frame_size, &info, &decoded_request,
                             &request_id, &reason) == SAGR_STATUS_SUCCESS &&
                         decoded_request.byte_count == 5672U &&
                         decoded_request.chunk_index == 2U,
                    "COMMIT carries image size and chunk count");
  request.chunk_index = 1U;
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 9U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "COMMIT rejects a chunk count inconsistent with image size");
  request.chunk_index = 2U;
  request.image_offset = 1U;
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 9U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "COMMIT rejects a nonzero image offset");
  request.image_offset = 0U;
  failures += expect(sagr_protocol_encode_code_object_request(
                         &info, 9U, &request, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS,
                    "COMMIT re-encodes with canonical prefix");
  memcpy(mutated, frame, sizeof(mutated));
  mutated[SAGR_WIRE_HEADER_BYTES + 36U + 3U] = 1U;
  sagr_protocol_recompute_frame_crc(mutated, sizeof(mutated));
  failures += expect(sagr_protocol_decode_code_object_request(
                         mutated, sizeof(mutated), &info, &decoded_request,
                         &request_id, &reason) != SAGR_STATUS_SUCCESS,
                    "COMMIT decoder rejects a mismatched chunk count");

  memset(&response, 0, sizeof(response));
  response.major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
  response.minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT;
  response.object_id = UINT64_C(9);
  response.generation = UINT64_C(10);
  response.accepted_offset = 5672U;
  response.accepted_count = 5672U;
  response.chunk_index = 2U;
  response.image_size = 5672U;
  response.segment_count = 3U;
  memcpy(response.image_sha256, digest, sizeof(digest));
  failures += expect(sagr_protocol_encode_code_object_response(
                         &info, 9U, &response, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS &&
                         sagr_protocol_decode_code_object_response(
                             frame, frame_size, &info, 9U, &decoded_response,
                             &wire_status, &reason) == SAGR_STATUS_SUCCESS &&
                         decoded_response.mapped_base_va == 0U &&
                         decoded_response.code_va == 0U && wire_status == 0,
                    "successful ACK preserves A1 zero-address boundary");
  response.mapped_base_va = 1U;
  failures += expect(sagr_protocol_encode_code_object_response(
                         &info, 9U, &response, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "ACK encoder rejects nonzero A1 mapping address");
  response.mapped_base_va = 0U;
  response.descriptor_va = 1U;
  failures += expect(sagr_protocol_encode_code_object_response(
                         &info, 9U, &response, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "ACK encoder rejects nonzero descriptor address");
  response.descriptor_va = 0U;
  response.code_va = 1U;
  failures += expect(sagr_protocol_encode_code_object_response(
                         &info, 9U, &response, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "ACK encoder rejects nonzero code address");
  response.code_va = 0U;
  response.kernarg_va = 1U;
  failures += expect(sagr_protocol_encode_code_object_response(
                         &info, 9U, &response, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "ACK encoder rejects nonzero kernarg address");
  response.kernarg_va = 0U;
  failures += expect(sagr_protocol_encode_code_object_response(
                         &info, 9U, &response, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS,
                    "ACK re-encodes after mapping address is cleared");
  memcpy(mutated, frame, sizeof(mutated));
  mutated[SAGR_WIRE_HEADER_BYTES + 140U] = 0xa5U;
  sagr_protocol_recompute_frame_crc(mutated, sizeof(mutated));
  failures += expect(sagr_protocol_decode_code_object_response(
                         mutated, sizeof(mutated), &info, 9U, &decoded_response,
                         &wire_status, &reason) != SAGR_STATUS_SUCCESS,
                    "ACK reserved0 at offset 140 is rejected");
  memcpy(mutated, frame, sizeof(mutated));
  mutated[SAGR_WIRE_HEADER_BYTES + 144U] = 0xa5U;
  sagr_protocol_recompute_frame_crc(mutated, sizeof(mutated));
  failures += expect(sagr_protocol_decode_code_object_response(
                         mutated, sizeof(mutated), &info, 9U, &decoded_response,
                         &wire_status, &reason) != SAGR_STATUS_SUCCESS,
                    "ACK reserved tail at offset 144 is rejected");
  memset(&response, 0, sizeof(response));
  response.major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
  response.minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_MALFORMED;
  response.opcode = SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK;
  response.error_code = SAGR_WIRE_STATUS_MALFORMED;
  failures += expect(sagr_protocol_encode_code_object_response(
                         &info, 10U, &response, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS,
                    "canonical failed ACK encodes");
  wire_status = -1;
  failures += expect(sagr_protocol_decode_code_object_response(
                         frame, frame_size, &info, 10U, &decoded_response,
                         &wire_status, &reason) == SAGR_STATUS_PROTOCOL_ERROR &&
                         wire_status == SAGR_WIRE_STATUS_MALFORMED &&
                         decoded_response.object_id == 0U,
                    "canonical failed ACK remains determinate");
  response.object_id = 1U;
  failures += expect(sagr_protocol_encode_code_object_response(
                         &info, 10U, &response, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_INVALID_ARGUMENT,
                    "failed ACK encoder rejects identity publication");
  response.object_id = 0U;
  failures += expect(sagr_protocol_encode_code_object_response(
                         &info, 10U, &response, frame, sizeof(frame),
                         &frame_size) == SAGR_STATUS_SUCCESS,
                    "failed ACK re-encodes after identity is cleared");
  memcpy(mutated, frame, sizeof(mutated));
  mutated[SAGR_WIRE_HEADER_BYTES + 23U] = 1U;
  sagr_protocol_recompute_frame_crc(mutated, sizeof(mutated));
  wire_status = 99;
  failures += expect(sagr_protocol_decode_code_object_response(
                         mutated, sizeof(mutated), &info, 10U,
                         &decoded_response, &wire_status, &reason) ==
                         SAGR_STATUS_PROTOCOL_ERROR &&
                         wire_status == -1,
                    "malformed failed ACK remains indeterminate");
  return failures == 0 ? 0 : 1;
}
