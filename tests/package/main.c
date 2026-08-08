/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/runtime.h>

int main(void) {
  sagr_instance_open_options_t options;
  if (sagr_abi_version() != SAGR_ABI_VERSION ||
      sagr_instance_open_options_init(&options, (uint32_t)sizeof(options)) !=
          SAGR_STATUS_SUCCESS ||
      options.minimum_version_major != 1 ||
      options.required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] !=
          SAGR_CAPABILITY_TOPOLOGY_MASK) {
    return 1;
  }
  return 0;
}
