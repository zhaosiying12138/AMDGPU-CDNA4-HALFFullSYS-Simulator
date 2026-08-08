/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stddef.h>

#include <self_amdgpu_runtime/runtime.h>

int main(void) {
  sagr_instance_open_options_t options;
  sagr_queue_create_options_t create_options;
  sagr_queue_operation_options_t operation_options;
  sagr_memory_allocate_options_t memory_allocate_options;
  sagr_memory_operation_options_t memory_operation_options;
  sagr_memory_info_t memory_info;
  sagr_memory_t memory = NULL;
  uint8_t byte = 0;
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
      options.minimum_version_major != 1 ||
      options.required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] !=
          SAGR_CAPABILITY_TOPOLOGY_MASK ||
      create_options.depth != 1 ||
      operation_options.cancel_fd != -1 ||
      SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST != UINT64_C(2) ||
      memory_allocate_options.alignment_bytes != SAGR_MEMORY_ALIGNMENT_4K ||
      memory_operation_options.cancel_fd != -1 ||
      SAGR_CAPABILITY_MEMORY_MASK != UINT64_C(4) ||
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
  return 0;
}
