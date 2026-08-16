/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "opencl_internal.h"

#include <limits.h>
#include <string.h>

cl_int sagr_cl_prepare_launch_geometry(
    cl_uint work_dimensions, const size_t *global_offset,
    const size_t *global_size, const size_t *local_size,
    uint32_t max_flat_workgroup_size, uint32_t wavefront_size,
    struct sagr_cl_launch_geometry *geometry) {
  uint64_t flat_local = 1U;
  uint64_t total_workgroups = 1U;
  cl_uint dimension;
  if (work_dimensions < 1U || work_dimensions > 3U) {
    return CL_INVALID_WORK_DIMENSION;
  }
  if (global_size == NULL) {
    return CL_INVALID_GLOBAL_WORK_SIZE;
  }
  if (local_size == NULL) {
    return CL_INVALID_WORK_GROUP_SIZE;
  }
  if (geometry == NULL) {
    return CL_INVALID_VALUE;
  }
  if (max_flat_workgroup_size == 0U || wavefront_size != 64U) {
    return CL_INVALID_PROGRAM_EXECUTABLE;
  }

  memset(geometry, 0, sizeof(*geometry));
  for (dimension = 0U; dimension < 3U; ++dimension) {
    geometry->global[dimension] = 1U;
    geometry->local[dimension] = 1U;
  }
  for (dimension = 0U; dimension < work_dimensions; ++dimension) {
    uint64_t rounded_extent;
    uint64_t groups;
    if (global_size[dimension] == 0U ||
        global_size[dimension] > (size_t)INT_MAX) {
      return CL_INVALID_GLOBAL_WORK_SIZE;
    }
    if (local_size[dimension] == 0U ||
        local_size[dimension] > (size_t)SAGR_CL_MAX_WORK_ITEM_SIZE) {
      return CL_INVALID_WORK_GROUP_SIZE;
    }
    if (global_size[dimension] % local_size[dimension] != 0U) {
      return CL_INVALID_WORK_GROUP_SIZE;
    }
    if (global_offset != NULL &&
        global_offset[dimension] > SIZE_MAX - global_size[dimension]) {
      return CL_INVALID_GLOBAL_OFFSET;
    }
    rounded_extent = (uint64_t)global_size[dimension] +
                     (uint64_t)local_size[dimension] - UINT64_C(1);
    if (rounded_extent > (uint64_t)INT_MAX) {
      return CL_INVALID_GLOBAL_WORK_SIZE;
    }
    groups = rounded_extent / (uint64_t)local_size[dimension];
    if (groups > (uint64_t)INT_MAX / total_workgroups) {
      return CL_INVALID_GLOBAL_WORK_SIZE;
    }
    total_workgroups *= groups;
    flat_local *= (uint64_t)local_size[dimension];
    if (flat_local > (uint64_t)SAGR_CL_MAX_WORK_GROUP_SIZE ||
        flat_local > (uint64_t)max_flat_workgroup_size) {
      return CL_INVALID_WORK_GROUP_SIZE;
    }
    geometry->global[dimension] = (uint64_t)global_size[dimension];
    geometry->local[dimension] = (uint64_t)local_size[dimension];
    geometry->offset[dimension] =
        global_offset == NULL ? 0U : (uint64_t)global_offset[dimension];
  }
  if (flat_local % (uint64_t)wavefront_size != 0U) {
    return CL_INVALID_WORK_GROUP_SIZE;
  }
  geometry->total_workgroups = total_workgroups;
  geometry->num_warps = (uint32_t)(flat_local / wavefront_size);
  geometry->dynamic_shared_memory_bytes = 0U;
  return CL_SUCCESS;
}
