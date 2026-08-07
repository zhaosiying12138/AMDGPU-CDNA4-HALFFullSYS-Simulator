/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/runtime.h>

int main(void) {
  return sagr_abi_version() == SAGR_ABI_VERSION ? 0 : 1;
}
