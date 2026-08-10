/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/runtime.h>

#include <string.h>

uint32_t sagr_abi_version(void) {
  return SAGR_ABI_VERSION;
}

const char *sagr_version_string(void) {
  return SAGR_VERSION_STRING;
}

const char *sagr_status_string(sagr_status_t status) {
  switch (status) {
    case SAGR_STATUS_SUCCESS:
      return "success";
    case SAGR_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case SAGR_STATUS_NOT_SUPPORTED:
      return "not supported";
    case SAGR_STATUS_INTERNAL_ERROR:
      return "internal error";
    case SAGR_STATUS_VERSION_MISMATCH:
      return "version mismatch";
    case SAGR_STATUS_CAPABILITY_MISMATCH:
      return "capability mismatch";
    case SAGR_STATUS_ENDPOINT_NOT_FOUND:
      return "endpoint not found";
    case SAGR_STATUS_INSTANCE_MISMATCH:
      return "instance mismatch";
    case SAGR_STATUS_TOPOLOGY_MISMATCH:
      return "topology mismatch";
    case SAGR_STATUS_PROTOCOL_ERROR:
      return "protocol error";
    case SAGR_STATUS_CHECKSUM_ERROR:
      return "checksum error";
    case SAGR_STATUS_TIMED_OUT:
      return "timed out";
    case SAGR_STATUS_UNAVAILABLE:
      return "unavailable";
    case SAGR_STATUS_CONNECTION_LOST:
      return "connection lost";
    case SAGR_STATUS_OUT_OF_RESOURCES:
      return "out of resources";
    case SAGR_STATUS_INVALID_HANDLE:
      return "invalid handle";
    case SAGR_STATUS_BUFFER_TOO_SMALL:
      return "buffer too small";
    case SAGR_STATUS_UNAUTHORIZED:
      return "unauthorized";
    case SAGR_STATUS_BUSY:
      return "busy";
    case SAGR_STATUS_CANCELLED:
      return "cancelled";
    default:
      return "unknown status";
  }
}

sagr_status_t sagr_instance_open_options_init(
    sagr_instance_open_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }

  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->minimum_version_major = 1;
  options->maximum_version_major = 1;
  options->open_timeout_ns = SAGR_DEFAULT_OPEN_TIMEOUT_NS;
  options->cancel_fd = -1;
  options->offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] =
      SAGR_CAPABILITY_TOPOLOGY_MASK;
  options->required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] =
      SAGR_CAPABILITY_TOPOLOGY_MASK;
  options->expected_rank = SAGR_INSTANCE_RANK_WILDCARD;

  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_queue_operation_options_init(
    sagr_queue_operation_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->timeout_ns = SAGR_DEFAULT_OPEN_TIMEOUT_NS;
  options->cancel_fd = -1;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_queue_create_options_init(
    sagr_queue_create_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->depth = 1;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_memory_allocate_options_init(
    sagr_memory_allocate_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->alignment_bytes = SAGR_MEMORY_ALIGNMENT_4K;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_memory_operation_options_init(
    sagr_memory_operation_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->timeout_ns = SAGR_DEFAULT_OPEN_TIMEOUT_NS;
  options->cancel_fd = -1;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_signal_create_options_init(
    sagr_signal_create_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_signal_operation_options_init(
    sagr_signal_operation_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->timeout_ns = SAGR_DEFAULT_OPEN_TIMEOUT_NS;
  options->cancel_fd = -1;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_pinned_dispatch_options_init(
    sagr_pinned_dispatch_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->fixture_id = SAGR_DISPATCH_FIXTURE_GFX950_XOR_U8_V1;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_generic_map_options_init(
    sagr_generic_map_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->version = SAGR_GENERIC_RUNTIME_API_VERSION;
  options->gfx_target = UINT32_C(950);
  options->relocation_count = 0U;
  options->descriptor_preload_dwords = 0U;
  options->page_size = SAGR_GENERIC_PAGE_SIZE_4K;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_generic_kernarg_allocate_options_init(
    sagr_generic_kernarg_allocate_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->version = SAGR_GENERIC_RUNTIME_API_VERSION;
  options->alignment_bytes = UINT64_C(8);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_generic_submit_options_init(
    sagr_generic_submit_options_t *options, uint32_t options_size) {
  if (options == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->version = SAGR_GENERIC_RUNTIME_API_VERSION;
  options->expected_signal_value_bits = UINT64_C(1);
  options->num_ctas = 1U;
  options->wavefront_size = UINT32_C(64);
  return SAGR_STATUS_SUCCESS;
}
