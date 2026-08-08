/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stddef.h>
#include <string.h>

#include <self_amdgpu_runtime/provider.h>
#include <self_amdgpu_runtime/runtime.h>

int main(void) {
  sagr_instance_open_options_t options;
  sagr_queue_create_options_t create_options;
  sagr_queue_operation_options_t operation_options;
  sagr_memory_allocate_options_t memory_allocate_options;
  sagr_memory_operation_options_t memory_operation_options;
  sagr_memory_info_t memory_info;
  sagr_pinned_dispatch_options_t dispatch_options;
  sagr_dispatch_ticket_t dispatch_ticket;
  sagr_dispatch_completion_t dispatch_completion;
  sagr_memory_t memory = NULL;
  sagr_provider_manifest_t provider_manifest;
  uint8_t byte = 0;
  if (sagr_provider_manifest(&provider_manifest,
                             (uint32_t)sizeof(provider_manifest)) !=
          SAGR_STATUS_SUCCESS ||
      provider_manifest.loader_entry_count !=
          SAGR_PROVIDER_LOADER_ENTRY_COUNT ||
      provider_manifest.target_loader_entry_count !=
          SAGR_PROVIDER_TARGET_LOADER_ENTRY_COUNT ||
      provider_manifest.direct_target_loader_entry_count !=
          SAGR_PROVIDER_DIRECT_TARGET_LOADER_ENTRY_COUNT ||
      strcmp(provider_manifest.authority_sha256,
             SAGR_PROVIDER_AUTHORITY_SHA256_HEX) != 0) {
    return 1;
  }
  if (sagr_abi_version() != SAGR_ABI_VERSION ||
      sagr_instance_open_options_init(&options, (uint32_t)sizeof(options)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_create_options_init(
          &create_options, (uint32_t)sizeof(create_options)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_operation_options_init(
          &operation_options, (uint32_t)sizeof(operation_options)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_memory_allocate_options_init(
          &memory_allocate_options,
          (uint32_t)sizeof(memory_allocate_options)) != SAGR_STATUS_SUCCESS ||
      sagr_memory_operation_options_init(
          &memory_operation_options,
          (uint32_t)sizeof(memory_operation_options)) != SAGR_STATUS_SUCCESS ||
      sagr_pinned_dispatch_options_init(
          &dispatch_options, (uint32_t)sizeof(dispatch_options)) !=
          SAGR_STATUS_SUCCESS ||
      options.minimum_version_major != 1 ||
      options.required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] !=
          SAGR_CAPABILITY_TOPOLOGY_MASK ||
      create_options.depth != 1 ||
      operation_options.cancel_fd != -1 ||
      SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST != UINT64_C(2) ||
      memory_allocate_options.alignment_bytes != SAGR_MEMORY_ALIGNMENT_4K ||
      memory_operation_options.cancel_fd != -1 ||
      SAGR_CAPABILITY_MEMORY_MASK != UINT64_C(4) ||
      SAGR_CAPABILITY_DISPATCH_MASK != UINT64_C(16) ||
      SAGR_DISPATCH_PACKET_CRC32C != UINT32_C(0x8a912d83) ||
      dispatch_options.fixture_id !=
          SAGR_DISPATCH_FIXTURE_GFX950_XOR_U8_V1 ||
      sagr_memory_allocate(NULL, &memory_allocate_options, NULL, &memory, NULL,
                           0, NULL, 0) != SAGR_STATUS_INVALID_ARGUMENT ||
      sagr_memory_get_info(NULL, &memory_info,
                           (uint32_t)sizeof(memory_info)) !=
          SAGR_STATUS_INVALID_HANDLE ||
      sagr_memory_copy_from_host(NULL, 0, &byte, 1, NULL, NULL, 0) !=
          SAGR_STATUS_INVALID_HANDLE ||
      sagr_memory_copy_to_host(NULL, 0, &byte, 1, NULL, NULL, 0) !=
          SAGR_STATUS_INVALID_HANDLE ||
      sagr_memory_free(&memory, NULL, NULL, 0) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  if (sagr_queue_submit_pinned_dispatch(
          NULL, NULL, NULL, NULL, &dispatch_options, NULL, &dispatch_ticket,
          (uint32_t)sizeof(dispatch_ticket), NULL, 0) !=
          SAGR_STATUS_INVALID_HANDLE ||
      sagr_queue_wait_pinned_dispatch(
          NULL, &dispatch_ticket, NULL, &dispatch_completion,
          (uint32_t)sizeof(dispatch_completion), NULL, 0) !=
          SAGR_STATUS_INVALID_HANDLE) {
    return 1;
  }
  return 0;
}
