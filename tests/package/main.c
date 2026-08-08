/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/runtime.h>

int main(void) {
  sagr_instance_open_options_t options;
  sagr_queue_create_options_t create_options;
  sagr_queue_operation_options_t operation_options;
  if (sagr_abi_version() != SAGR_ABI_VERSION ||
      sagr_instance_open_options_init(&options, (uint32_t)sizeof(options)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_create_options_init(
          &create_options, (uint32_t)sizeof(create_options)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_operation_options_init(
          &operation_options, (uint32_t)sizeof(operation_options)) !=
          SAGR_STATUS_SUCCESS ||
      options.minimum_version_major != 1 ||
      options.required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] !=
          SAGR_CAPABILITY_TOPOLOGY_MASK ||
      create_options.depth != 1 ||
      operation_options.cancel_fd != -1 ||
      SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST != UINT64_C(2)) {
    return 1;
  }
  return 0;
}
