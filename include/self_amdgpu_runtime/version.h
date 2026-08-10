/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_VERSION_H
#define SELF_AMDGPU_RUNTIME_VERSION_H

#include <stdint.h>

#define SAGR_VERSION_MAJOR 0u
#define SAGR_VERSION_MINOR 6u
#define SAGR_VERSION_PATCH 0u
#define SAGR_VERSION_STRING "0.6.0"

#define SAGR_ABI_VERSION_MAJOR 1u
#define SAGR_ABI_VERSION_MINOR 6u

#define SAGR_ABI_VERSION_ENCODE(major, minor) \
  ((((uint32_t)(major)) << 16u) | ((uint32_t)(minor)))
#define SAGR_ABI_VERSION \
  SAGR_ABI_VERSION_ENCODE(SAGR_ABI_VERSION_MAJOR, SAGR_ABI_VERSION_MINOR)

#define SAGR_ABI_VERSION_DECODE_MAJOR(version) \
  ((uint32_t)(version) >> 16u)
#define SAGR_ABI_VERSION_DECODE_MINOR(version) \
  ((uint32_t)(version) & UINT32_C(0xffff))

#endif
