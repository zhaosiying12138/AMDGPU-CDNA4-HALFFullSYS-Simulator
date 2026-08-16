/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <self_amdgpu_runtime/kmt_shim.h>
#include <self_amdgpu_runtime/provider.h>

#include "provider_internal.h"
#include "transport_internal.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

static int bytes_are_zero(const uint8_t *bytes, size_t size) {
  uint8_t combined = 0;
  size_t index;
  for (index = 0; index < size; ++index) {
    combined = (uint8_t)(combined | bytes[index]);
  }
  return combined == 0;
}

static int operation_valid(uint16_t operation) {
  return sagr_bridge_kmt_operation_valid(operation);
}

static int hsakmt_status_valid(uint32_t status) {
  switch (status) {
    case SAGR_PROVIDER_HSAKMT_STATUS_SUCCESS:
    case SAGR_PROVIDER_HSAKMT_STATUS_ERROR:
    case SAGR_PROVIDER_HSAKMT_STATUS_DRIVER_MISMATCH:
    case SAGR_PROVIDER_HSAKMT_STATUS_INVALID_PARAMETER:
    case SAGR_PROVIDER_HSAKMT_STATUS_INVALID_HANDLE:
    case SAGR_PROVIDER_HSAKMT_STATUS_INVALID_NODE_UNIT:
    case SAGR_PROVIDER_HSAKMT_STATUS_NO_MEMORY:
    case SAGR_PROVIDER_HSAKMT_STATUS_BUFFER_TOO_SMALL:
    case SAGR_PROVIDER_HSAKMT_STATUS_NOT_IMPLEMENTED:
    case SAGR_PROVIDER_HSAKMT_STATUS_NOT_SUPPORTED:
    case SAGR_PROVIDER_HSAKMT_STATUS_UNAVAILABLE:
    case SAGR_PROVIDER_HSAKMT_STATUS_OUT_OF_RESOURCES:
    case SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED:
    case SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR:
    case SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_ALREADY_OPENED:
    case SAGR_PROVIDER_HSAKMT_STATUS_HSAMMU_UNAVAILABLE:
    case SAGR_PROVIDER_HSAKMT_STATUS_WAIT_FAILURE:
    case SAGR_PROVIDER_HSAKMT_STATUS_WAIT_TIMEOUT:
    case SAGR_PROVIDER_HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED:
    case SAGR_PROVIDER_HSAKMT_STATUS_MEMORY_NOT_REGISTERED:
    case SAGR_PROVIDER_HSAKMT_STATUS_MEMORY_ALIGNMENT:
      return 1;
    default:
      return 0;
  }
}

static int words_zero_outside(const uint32_t words[8], uint16_t operation,
                              int result_words) {
  uint32_t allowed_mask;
  uint32_t index;
  if (!sagr_bridge_kmt_operation_valid(operation))
    return 0;
  allowed_mask = result_words
                     ? sagr_bridge_kmt_result_word_mask(operation)
                     : sagr_bridge_kmt_request_word_mask(operation);
  for (index = 0; index < SAGR_KMT_ARGUMENT_WORD_COUNT; ++index) {
    if ((allowed_mask & (UINT32_C(1) << index)) == 0 && words[index] != 0) {
      return 0;
    }
  }
  return 1;
}

static sagr_kmt_status_t runtime_to_kmt(sagr_status_t status) {
  switch (status) {
    case SAGR_STATUS_SUCCESS:
      return SAGR_KMT_STATUS_SUCCESS;
    case SAGR_STATUS_INVALID_ARGUMENT:
      return SAGR_KMT_STATUS_INVALID_PARAMETER;
    case SAGR_STATUS_INVALID_HANDLE:
    case SAGR_STATUS_INSTANCE_MISMATCH:
      return SAGR_KMT_STATUS_INVALID_HANDLE;
    case SAGR_STATUS_BUFFER_TOO_SMALL:
      return SAGR_KMT_STATUS_BUFFER_TOO_SMALL;
    case SAGR_STATUS_NOT_SUPPORTED:
    case SAGR_STATUS_CAPABILITY_MISMATCH:
      return SAGR_KMT_STATUS_NOT_SUPPORTED;
    case SAGR_STATUS_TIMED_OUT:
      return SAGR_KMT_STATUS_WAIT_TIMEOUT;
    case SAGR_STATUS_OUT_OF_RESOURCES:
      return SAGR_KMT_STATUS_OUT_OF_RESOURCES;
    case SAGR_STATUS_UNAVAILABLE:
    case SAGR_STATUS_ENDPOINT_NOT_FOUND:
      return SAGR_KMT_STATUS_UNAVAILABLE;
    case SAGR_STATUS_CONNECTION_LOST:
    case SAGR_STATUS_PROTOCOL_ERROR:
    case SAGR_STATUS_CHECKSUM_ERROR:
    case SAGR_STATUS_INTERNAL_ERROR:
    case SAGR_STATUS_VERSION_MISMATCH:
    case SAGR_STATUS_TOPOLOGY_MISMATCH:
    case SAGR_STATUS_UNAUTHORIZED:
    case SAGR_STATUS_BUSY:
    case SAGR_STATUS_CANCELLED:
    default:
      return SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR;
  }
}

static sagr_kmt_status_t fail_local(sagr_error_info_t *error,
                                    uint32_t error_size,
                                    sagr_kmt_status_t status,
                                    const char *message) {
  if (error != NULL && error_size >= sizeof(*error)) {
    memset(error, 0, sizeof(*error));
    error->struct_size = (uint32_t)sizeof(*error);
    error->status = sagr_provider_status_to_runtime(status);
    error->wire_status = -1;
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return status;
}

static void put_word_u64(uint32_t *words, uint32_t index, uint64_t value) {
  words[index] = (uint32_t)(value >> 32);
  words[index + 1U] = (uint32_t)value;
}

static void put_buffer_u64(uint8_t *buffer, uint32_t offset, uint64_t value) {
  uint32_t index;
  for (index = 0U; index < 8U; ++index) {
    buffer[offset + index] = (uint8_t)(value >> (56U - index * 8U));
  }
}

static uint64_t get_word_u64(const uint32_t *words, uint32_t index) {
  return ((uint64_t)words[index] << 32) | (uint64_t)words[index + 1U];
}

static uint16_t get_be_u16(const uint8_t *bytes) {
  return (uint16_t)(((uint16_t)bytes[0] << 8) | bytes[1]);
}

static uint32_t get_be_u32(const uint8_t *bytes) {
  return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
         ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}

static uint64_t get_be_u64(const uint8_t *bytes) {
  return ((uint64_t)get_be_u32(bytes) << 32) | get_be_u32(bytes + 4);
}

static sagr_kmt_status_t validate_error_buffer(sagr_error_info_t *error,
                                               uint32_t error_size) {
  if ((error == NULL && error_size != 0U) ||
      (error != NULL && error_size < sizeof(*error))) {
    return SAGR_KMT_STATUS_INVALID_PARAMETER;
  }
  return SAGR_KMT_STATUS_SUCCESS;
}

static sagr_kmt_status_t validate_call_options(
    const sagr_kmt_call_options_t *options) {
  if (options == NULL) {
    return SAGR_KMT_STATUS_SUCCESS;
  }
  if (options->struct_size < sizeof(*options) || options->flags != 0 ||
      options->cancel_fd < -1 || options->reserved0 != 0 ||
      !bytes_are_zero(options->reserved, sizeof(options->reserved))) {
    return SAGR_KMT_STATUS_INVALID_PARAMETER;
  }
  return SAGR_KMT_STATUS_SUCCESS;
}

static sagr_kmt_status_t prepare_request(
    sagr_provider_t *provider, uint16_t operation,
    const sagr_kmt_call_options_t *options, sagr_kmt_envelope_request_t *request,
    sagr_error_info_t *error, uint32_t error_size) {
  sagr_kmt_status_t status;
  if (!sagr_provider_is_valid(provider)) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_HANDLE,
                      "provider handle is not open");
  }
  status = validate_error_buffer(error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  status = validate_call_options(options);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return fail_local(error, error_size, status, "invalid KMT call options");
  }
  memset(request, 0, sizeof(*request));
  request->major = SAGR_KMT_PROTOCOL_MAJOR;
  request->minor = SAGR_KMT_PROTOCOL_MINOR;
  request->operation = operation;
  request->operation_sequence = sagr_provider_next_kmt_sequence(provider);
  if (request->operation_sequence == 0) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_HANDLE,
                      "provider handle is not open");
  }
  return SAGR_KMT_STATUS_SUCCESS;
}

static sagr_kmt_status_t validate_kfd_handle(
    const sagr_kmt_handle_t *handle, sagr_error_info_t *error,
    uint32_t error_size) {
  if (handle == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "KMT handle pointer is null");
  }
  if (handle->owner_id == 0 || handle->owner_generation == 0 ||
      handle->object_id != 0 || handle->object_generation != 0) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_HANDLE,
                      "KFD owner handle is invalid");
  }
  return SAGR_KMT_STATUS_SUCCESS;
}

static sagr_kmt_status_t validate_resource_handle(
    const sagr_kmt_handle_t *owner, const sagr_kmt_handle_t *resource,
    sagr_error_info_t *error, uint32_t error_size) {
  if (resource == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "resource handle pointer is null");
  }
  if (resource->owner_id == 0 || resource->owner_generation == 0 ||
      resource->object_id == 0 || resource->object_generation == 0 ||
      resource->owner_id != owner->owner_id ||
      resource->owner_generation != owner->owner_generation) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_HANDLE,
                      "resource owner or generation is invalid");
  }
  return SAGR_KMT_STATUS_SUCCESS;
}

static void request_owner(sagr_kmt_envelope_request_t *request,
                          const sagr_kmt_handle_t *handle) {
  request->owner_id = handle->owner_id;
  request->owner_generation = handle->owner_generation;
}

static void request_object(sagr_kmt_envelope_request_t *request,
                           const sagr_kmt_handle_t *handle) {
  request->object_id = handle->object_id;
  request->object_generation = handle->object_generation;
}

static sagr_kmt_status_t exchange(
    sagr_provider_t *provider, const sagr_kmt_envelope_request_t *request,
    const sagr_kmt_call_options_t *options, sagr_kmt_envelope_result_t *result,
    sagr_error_info_t *error, uint32_t error_size) {
  int32_t wire_status = -1;
  sagr_status_t runtime_status;
  sagr_kmt_status_t status;
  memset(result, 0, sizeof(*result));
  runtime_status = sagr_transport_kmt_exchange(
      sagr_provider_transport_instance(provider), request, options, result,
      &wire_status, error, error_size);
  if (runtime_status != SAGR_STATUS_SUCCESS) {
    return runtime_to_kmt(runtime_status);
  }
  status = sagr_kmt_envelope_result_validate(request, result);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "noncanonical KMT result envelope");
  }
  if (result->wire_status != SAGR_WIRE_STATUS_OK) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "KMT result carries a transport failure");
  }
  status = (sagr_kmt_status_t)result->status;
  if (error != NULL && error_size >= sizeof(*error)) {
    error->status = sagr_provider_status_to_runtime(status);
    error->wire_status = wire_status;
  }
  return status;
}

static sagr_kmt_status_t exchange_with_descriptor(
    sagr_provider_t *provider, const sagr_kmt_envelope_request_t *request,
    int descriptor, const sagr_kmt_call_options_t *options,
    sagr_kmt_envelope_result_t *result, sagr_error_info_t *error,
    uint32_t error_size) {
  int32_t wire_status = -1;
  sagr_status_t runtime_status;
  sagr_kmt_status_t status;
  memset(result, 0, sizeof(*result));
  runtime_status = sagr_transport_kmt_exchange_with_descriptor(
      sagr_provider_transport_instance(provider), request, descriptor, options,
      result, &wire_status, error, error_size);
  if (runtime_status != SAGR_STATUS_SUCCESS) {
    return runtime_to_kmt(runtime_status);
  }
  status = sagr_kmt_envelope_result_validate(request, result);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "noncanonical KMT backing result envelope");
  }
  if (result->wire_status != SAGR_WIRE_STATUS_OK) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "KMT backing result carries a transport failure");
  }
  status = (sagr_kmt_status_t)result->status;
  if (error != NULL && error_size >= sizeof(*error)) {
    error->status = sagr_provider_status_to_runtime(status);
    error->wire_status = wire_status;
  }
  return status;
}

sagr_kmt_status_t sagr_kmt_call_options_init(sagr_kmt_call_options_t *options,
                                              uint32_t options_size) {
  if (options == NULL) {
    return SAGR_KMT_STATUS_INVALID_PARAMETER;
  }
  if (options_size < sizeof(*options)) {
    if (options_size >= sizeof(options->struct_size)) {
      options->struct_size = (uint32_t)sizeof(*options);
    }
    return SAGR_KMT_STATUS_BUFFER_TOO_SMALL;
  }
  memset(options, 0, options_size);
  options->struct_size = options_size;
  options->timeout_ns = SAGR_DEFAULT_OPEN_TIMEOUT_NS;
  options->cancel_fd = -1;
  return SAGR_KMT_STATUS_SUCCESS;
}

sagr_kmt_status_t sagr_kmt_envelope_request_init(
    sagr_kmt_envelope_request_t *request, uint32_t request_size,
    uint16_t operation) {
  if (request == NULL || !operation_valid(operation)) {
    return SAGR_KMT_STATUS_INVALID_PARAMETER;
  }
  if (request_size < sizeof(*request)) {
    return SAGR_KMT_STATUS_BUFFER_TOO_SMALL;
  }
  memset(request, 0, request_size);
  request->major = SAGR_KMT_PROTOCOL_MAJOR;
  request->minor = SAGR_KMT_PROTOCOL_MINOR;
  request->operation = operation;
  return SAGR_KMT_STATUS_SUCCESS;
}

sagr_kmt_status_t sagr_kmt_envelope_result_validate(
    const sagr_kmt_envelope_request_t *request,
    const sagr_kmt_envelope_result_t *result) {
  const int opens_owner =
      request != NULL && request->operation == SAGR_KMT_OP_OPEN_KFD;
  const int creates_object =
      request != NULL &&
      (request->operation == SAGR_KMT_OP_TOPOLOGY_SNAPSHOT ||
       request->operation == SAGR_KMT_OP_ALLOC_MEMORY ||
       request->operation == SAGR_KMT_OP_ALLOC_MEMORY_OF_GPU ||
       request->operation == SAGR_KMT_OP_QUEUE_CREATE ||
       request->operation == SAGR_KMT_OP_EVENT_CREATE);
  if (request == NULL || result == NULL ||
      result->major != SAGR_KMT_PROTOCOL_MAJOR ||
      result->minor != SAGR_KMT_PROTOCOL_MINOR ||
      result->operation != request->operation || result->flags != 0 ||
      result->operation_sequence != request->operation_sequence ||
      result->wire_status != SAGR_WIRE_STATUS_OK ||
      (!opens_owner &&
       (result->owner_id != request->owner_id ||
        result->owner_generation != request->owner_generation)) ||
      result->auxiliary_id != request->auxiliary_id ||
      result->auxiliary_generation != request->auxiliary_generation ||
      !words_zero_outside(request->argument_words, request->operation, 0) ||
      !words_zero_outside(result->result_words, result->operation, 1) ||
      (!creates_object &&
       (result->object_id != request->object_id ||
        result->object_generation != request->object_generation)) ||
      !hsakmt_status_valid(result->status) ||
      result->buffer_bytes > SAGR_KMT_BUFFER_BYTES ||
      (result->buffer_bytes == 0 && result->buffer_crc32c != 0) ||
      (result->buffer_bytes != 0 &&
       result->buffer_crc32c !=
           sagr_crc32c(result->buffer, result->buffer_bytes)) ||
      !bytes_are_zero(result->buffer + result->buffer_bytes,
                      SAGR_KMT_BUFFER_BYTES - result->buffer_bytes) ||
      !bytes_are_zero(result->reserved, sizeof(result->reserved))) {
    return SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR;
  }
  if (result->status == SAGR_KMT_STATUS_SUCCESS) {
    if (opens_owner &&
        (result->owner_id == 0 || result->owner_generation == 0 ||
         result->object_id != 0 || result->object_generation != 0)) {
      return SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR;
    }
    if (creates_object &&
        (result->object_id == 0 || result->object_generation == 0)) {
      return SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR;
    }
  } else {
    uint32_t index;
    if ((opens_owner &&
         (result->owner_id != 0 || result->owner_generation != 0)) ||
        (creates_object &&
         (result->object_id != 0 || result->object_generation != 0)) ||
        result->buffer_bytes != 0 || result->buffer_crc32c != 0) {
      return SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR;
    }
    for (index = 0; index < SAGR_KMT_ARGUMENT_WORD_COUNT; ++index) {
      if (result->result_words[index] != 0) {
        return SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR;
      }
    }
  }
  return SAGR_KMT_STATUS_SUCCESS;
}

sagr_kmt_status_t sagr_kmt_alloc_memory_of_gpu(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint64_t virtual_address, uint64_t size_bytes, uint32_t gpu_id,
    uint32_t memory_flags, uint64_t mmap_offset,
    sagr_kmt_handle_t *out_memory, uint64_t *out_mmap_offset,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_handle_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_ALLOC_MEMORY_OF_GPU, options, &request, error,
      error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS)
    status = validate_kfd_handle(handle, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS)
    return status;
  if (gpu_id == 0U || size_bytes == 0U ||
      virtual_address > UINT64_MAX - size_bytes ||
      (virtual_address != 0U && (virtual_address & UINT64_C(0xfff)) != 0U) ||
      (mmap_offset != 0U && (mmap_offset & UINT64_C(0xfff)) != 0U) ||
      (mmap_offset != 0U && mmap_offset > UINT64_MAX - size_bytes) ||
      out_memory == NULL || out_mmap_offset == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid GPU memory allocation carriers");
  }
  request_owner(&request, handle);
  put_word_u64(request.argument_words, 0, virtual_address);
  put_word_u64(request.argument_words, 2, size_bytes);
  put_word_u64(request.argument_words, 4, mmap_offset);
  request.argument_words[6] = gpu_id;
  request.argument_words[7] = memory_flags;
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS)
    return status;
  if (result.object_id == 0U || result.object_generation == 0U)
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "GPU allocation returned an empty handle");
  committed.owner_id = result.owner_id;
  committed.owner_generation = result.owner_generation;
  committed.object_id = result.object_id;
  committed.object_generation = result.object_generation;
  *out_memory = committed;
  *out_mmap_offset = get_word_u64(result.result_words, 0);
  return SAGR_KMT_STATUS_SUCCESS;
}

sagr_kmt_status_t sagr_kmt_free_memory_of_gpu(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_FREE_MEMORY_OF_GPU, options, &request, error,
      error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS)
    status = validate_kfd_handle(handle, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS)
    status = validate_resource_handle(handle, memory, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS)
    return status;
  request_owner(&request, memory);
  request_object(&request, memory);
  return exchange(provider, &request, options, &result, error, error_size);
}

sagr_kmt_status_t sagr_kmt_map_memory_to_gpu(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const uint32_t *gpu_ids,
    uint32_t gpu_count, uint32_t *out_success,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_MAP_MEMORY_TO_GPU, options, &request, error,
      error_size);
  uint32_t index;
  if (status == SAGR_KMT_STATUS_SUCCESS)
    status = validate_kfd_handle(handle, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS)
    status = validate_resource_handle(handle, memory, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS)
    return status;
  if (gpu_ids == NULL || gpu_count == 0U || gpu_count > 16U ||
      out_success == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid GPU mapping carriers");
  }
  request_owner(&request, memory);
  request_object(&request, memory);
  request.argument_words[0] = gpu_count;
  request.argument_words[1] = 0;
  request.buffer_bytes = gpu_count * sizeof(uint32_t);
  for (index = 0U; index < gpu_count; ++index) {
    request.buffer[index * 4U] = (uint8_t)(gpu_ids[index] >> 24);
    request.buffer[index * 4U + 1U] = (uint8_t)(gpu_ids[index] >> 16);
    request.buffer[index * 4U + 2U] = (uint8_t)(gpu_ids[index] >> 8);
    request.buffer[index * 4U + 3U] = (uint8_t)gpu_ids[index];
  }
  request.buffer_crc32c = sagr_crc32c(request.buffer, request.buffer_bytes);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS)
    return status;
  if (result.result_words[0] > gpu_count)
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "invalid GPU mapping result count");
  *out_success = result.result_words[0];
  return SAGR_KMT_STATUS_SUCCESS;
}

sagr_kmt_status_t sagr_kmt_unmap_memory_from_gpu(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const uint32_t *gpu_ids,
    uint32_t gpu_count, uint32_t *out_success,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_UNMAP_MEMORY_FROM_GPU, options, &request, error,
      error_size);
  uint32_t index;
  if (status == SAGR_KMT_STATUS_SUCCESS)
    status = validate_kfd_handle(handle, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS)
    status = validate_resource_handle(handle, memory, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS)
    return status;
  if (gpu_ids == NULL || gpu_count == 0U || gpu_count > 16U ||
      out_success == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid GPU unmapping carriers");
  }
  request_owner(&request, memory);
  request_object(&request, memory);
  request.argument_words[0] = gpu_count;
  request.buffer_bytes = gpu_count * sizeof(uint32_t);
  for (index = 0U; index < gpu_count; ++index) {
    request.buffer[index * 4U] = (uint8_t)(gpu_ids[index] >> 24);
    request.buffer[index * 4U + 1U] = (uint8_t)(gpu_ids[index] >> 16);
    request.buffer[index * 4U + 2U] = (uint8_t)(gpu_ids[index] >> 8);
    request.buffer[index * 4U + 3U] = (uint8_t)gpu_ids[index];
  }
  request.buffer_crc32c = sagr_crc32c(request.buffer, request.buffer_bytes);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS)
    return status;
  if (result.result_words[0] > gpu_count)
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "invalid GPU unmapping result count");
  *out_success = result.result_words[0];
  return SAGR_KMT_STATUS_SUCCESS;
}

sagr_kmt_status_t sagr_kmt_set_scratch_backing_va(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t gpu_id, uint64_t va_addr,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_SET_SCRATCH_BACKING_VA, options, &request, error,
      error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS)
    status = validate_kfd_handle(handle, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS)
    return status;
  if (gpu_id == 0U || va_addr == 0U || (va_addr & UINT64_C(0xffff)) != 0U)
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid scratch backing address");
  request_owner(&request, handle);
  request.argument_words[0] = gpu_id;
  put_word_u64(request.argument_words, 1, va_addr);
  return exchange(provider, &request, options, &result, error, error_size);
}

sagr_kmt_status_t sagr_kmt_export_backing(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    int backing_fd, uint64_t backing_bytes, uint32_t page_bytes,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  struct stat metadata;
  int descriptor_flags;
  int status_flags;
  int seals;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_EXPORT_BACKING, options, &request, error,
      error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS)
    status = validate_kfd_handle(handle, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS)
    return status;
  descriptor_flags = backing_fd >= 0 ? fcntl(backing_fd, F_GETFD) : -1;
  status_flags = backing_fd >= 0 ? fcntl(backing_fd, F_GETFL) : -1;
  seals = backing_fd >= 0 ? fcntl(backing_fd, F_GET_SEALS) : -1;
  if (backing_fd < 0 || descriptor_flags < 0 ||
      (descriptor_flags & FD_CLOEXEC) == 0 || status_flags < 0 ||
      (status_flags & O_ACCMODE) != O_RDWR || fstat(backing_fd, &metadata) != 0 ||
      !S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
      (uint64_t)metadata.st_size != backing_bytes || backing_bytes == 0U ||
      page_bytes < UINT32_C(4096) ||
      (page_bytes & (page_bytes - UINT32_C(1))) != 0U ||
      backing_bytes % page_bytes != 0U || seals < 0 ||
      (seals & (F_SEAL_SHRINK | F_SEAL_GROW)) !=
          (F_SEAL_SHRINK | F_SEAL_GROW)) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid shared KMT backing descriptor");
  }
  request_owner(&request, handle);
  put_word_u64(request.argument_words, 0, backing_bytes);
  request.argument_words[2] = page_bytes;
  return exchange_with_descriptor(provider, &request, backing_fd, options,
                                  &result, error, error_size);
}

sagr_kmt_status_t sagr_kmt_open_kfd(
    sagr_provider_t *provider, sagr_kmt_handle_t *out_handle,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_handle_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_OPEN_KFD, options, &request, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (out_handle == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "open KFD output handle is null");
  }
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  committed.owner_id = result.owner_id;
  committed.owner_generation = result.owner_generation;
  committed.object_id = result.object_id;
  committed.object_generation = result.object_generation;
  *out_handle = committed;
  return status;
}

sagr_kmt_status_t sagr_kmt_close_kfd(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_CLOSE_KFD, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  request_owner(&request, handle);
  return exchange(provider, &request, options, &result, error, error_size);
}

sagr_kmt_status_t sagr_kmt_get_version(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    sagr_kmt_version_t *out_version, uint32_t version_size,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_version_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_GET_VERSION, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (out_version == NULL || version_size < sizeof(*out_version)) {
    return fail_local(error, error_size,
                      out_version == NULL ? SAGR_KMT_STATUS_INVALID_PARAMETER
                                          : SAGR_KMT_STATUS_BUFFER_TOO_SMALL,
                      "version output buffer is invalid");
  }
  request_owner(&request, handle);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  memset(&committed, 0, sizeof(committed));
  committed.struct_size = (uint32_t)sizeof(committed);
  committed.major = result.result_words[0];
  committed.minor = result.result_words[1];
  committed.patch = result.result_words[2];
  committed.flags = result.result_words[3];
  memcpy(out_version, &committed, sizeof(committed));
  return status;
}

sagr_kmt_status_t sagr_kmt_get_clock_counters(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t gpu_id, sagr_kmt_clock_counters_t *out_counters,
    uint32_t counters_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_clock_counters_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_GET_CLOCK_COUNTERS, options, &request, error,
      error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (gpu_id == 0U || out_counters == NULL ||
      counters_size < sizeof(*out_counters)) {
    return fail_local(
        error, error_size,
        out_counters == NULL || gpu_id == 0U
            ? SAGR_KMT_STATUS_INVALID_PARAMETER
            : SAGR_KMT_STATUS_BUFFER_TOO_SMALL,
        "clock-counter request or output buffer is invalid");
  }
  request_owner(&request, handle);
  request.argument_words[0] = gpu_id;
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  memset(&committed, 0, sizeof(committed));
  committed.struct_size = (uint32_t)sizeof(committed);
  committed.gpu_clock_counter = get_word_u64(result.result_words, 0U);
  committed.cpu_clock_counter = get_word_u64(result.result_words, 2U);
  committed.system_clock_counter = get_word_u64(result.result_words, 4U);
  committed.system_clock_frequency_hz =
      get_word_u64(result.result_words, 6U);
  if (committed.gpu_clock_counter == 0U ||
      committed.cpu_clock_counter == 0U ||
      committed.system_clock_counter == 0U ||
      committed.gpu_clock_counter != committed.cpu_clock_counter ||
      committed.gpu_clock_counter != committed.system_clock_counter ||
      committed.system_clock_frequency_hz !=
          SAGR_BRIDGE_KMT_CLOCK_FREQUENCY_HZ) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "noncanonical simulated clock-counter result");
  }
  memcpy(out_counters, &committed, sizeof(committed));
  return status;
}

sagr_kmt_status_t sagr_kmt_topology_snapshot(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    sagr_kmt_topology_t *out_topology, uint32_t topology_size,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_topology_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_TOPOLOGY_SNAPSHOT, options, &request, error,
      error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (out_topology == NULL || topology_size < sizeof(*out_topology)) {
    return fail_local(error, error_size,
                      out_topology == NULL ? SAGR_KMT_STATUS_INVALID_PARAMETER
                                           : SAGR_KMT_STATUS_BUFFER_TOO_SMALL,
                      "topology output buffer is invalid");
  }
  request_owner(&request, handle);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (result.object_id != 1U ||
      result.object_generation != get_word_u64(result.result_words, 4) ||
      result.buffer_bytes != 64U || result.result_words[0] != 1U ||
      result.result_words[1] != 1U || result.result_words[2] != 0U ||
      result.result_words[3] != 0U || get_be_u16(result.buffer) != 1U ||
      get_be_u16(result.buffer + 2) != 0U ||
      get_be_u32(result.buffer + 4) != result.result_words[0] ||
      get_be_u64(result.buffer + 8) !=
          get_word_u64(result.result_words, 4) ||
      get_be_u32(result.buffer + 16) != 950U ||
      get_be_u32(result.buffer + 20) != 1U ||
      get_be_u32(result.buffer + 24) != 64U ||
      get_be_u32(result.buffer + 28) != 65536U ||
      get_be_u32(result.buffer + 32) != 48U ||
      get_be_u32(result.buffer + 36) != 8U ||
      get_be_u32(result.buffer + 40) != 1024U ||
      get_be_u32(result.buffer + 44) != 1024U ||
      get_be_u16(result.buffer + 48) != 1U ||
      get_be_u16(result.buffer + 50) != 1U ||
      !bytes_are_zero(result.buffer + 52, 12U)) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "noncanonical topology snapshot record");
  }
  memset(&committed, 0, sizeof(committed));
  committed.struct_size = (uint32_t)sizeof(committed);
  committed.snapshot_major = get_be_u16(result.buffer);
  committed.snapshot_minor = get_be_u16(result.buffer + 2);
  committed.model_major = get_be_u16(result.buffer + 48);
  committed.model_minor = get_be_u16(result.buffer + 50);
  committed.node_count = result.result_words[0];
  committed.gpu_node_count = result.result_words[1];
  committed.cpu_node_count = result.result_words[2];
  committed.gfx_target_code = get_be_u32(result.buffer + 16);
  committed.compute_units = get_be_u32(result.buffer + 20);
  committed.wavefront_size = get_be_u32(result.buffer + 24);
  committed.page_size = get_be_u32(result.buffer + 28);
  committed.va_bits = get_be_u32(result.buffer + 32);
  committed.maximum_queues = get_be_u32(result.buffer + 36);
  committed.maximum_allocations = get_be_u32(result.buffer + 40);
  committed.maximum_events = get_be_u32(result.buffer + 44);
  committed.topology_generation = get_word_u64(result.result_words, 4);
  memcpy(out_topology, &committed, sizeof(committed));
  return status;
}

sagr_kmt_status_t sagr_kmt_process_apertures(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t start_index, sagr_kmt_process_aperture_t *out_apertures,
    uint32_t capacity, uint32_t *out_returned, uint32_t *out_total,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_process_aperture_t committed[SAGR_KMT_PROCESS_APERTURES_PER_PAGE];
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_PROCESS_APERTURES, options, &request, error,
      error_size);
  uint32_t returned;
  uint32_t total;
  uint32_t index;
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (capacity > SAGR_KMT_PROCESS_APERTURES_PER_PAGE ||
      (capacity != 0U && out_apertures == NULL) || out_returned == NULL ||
      out_total == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid process aperture page carriers");
  }
  request_owner(&request, handle);
  request.argument_words[0] = start_index;
  request.argument_words[1] = capacity;
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  returned = result.result_words[1];
  total = result.result_words[2];
  if (result.result_words[0] != start_index || start_index > total ||
      returned > capacity || returned > total - start_index ||
      returned != (capacity < total - start_index ? capacity
                                                   : total - start_index) ||
      result.buffer_bytes !=
          returned * SAGR_KMT_PROCESS_APERTURE_WIRE_BYTES) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "noncanonical process aperture page");
  }
  memset(committed, 0, sizeof(committed));
  for (index = 0U; index < returned; ++index) {
    const uint8_t *record =
        result.buffer + index * SAGR_KMT_PROCESS_APERTURE_WIRE_BYTES;
    sagr_kmt_process_aperture_t *entry = &committed[index];
    entry->lds_base = get_be_u64(record);
    entry->lds_limit = get_be_u64(record + 8);
    entry->scratch_base = get_be_u64(record + 16);
    entry->scratch_limit = get_be_u64(record + 24);
    entry->gpuvm_base = get_be_u64(record + 32);
    entry->gpuvm_limit = get_be_u64(record + 40);
    entry->gpu_id = get_be_u32(record + 48);
    entry->reserved0 = get_be_u32(record + 52);
    if (entry->gpu_id == 0U || entry->reserved0 != 0U ||
        entry->lds_base > entry->lds_limit ||
        entry->scratch_base > entry->scratch_limit ||
        entry->gpuvm_base > entry->gpuvm_limit) {
      return fail_local(error, error_size,
                        SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                        "invalid process aperture record");
    }
  }
  if (returned != 0U) {
    memcpy(out_apertures, committed,
           (size_t)returned * sizeof(committed[0]));
  }
  *out_returned = returned;
  *out_total = total;
  return SAGR_KMT_STATUS_SUCCESS;
}

sagr_kmt_status_t sagr_kmt_acquire_vm(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t gpu_id, uint32_t render_minor,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_ACQUIRE_VM, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (gpu_id == 0U || render_minor < 128U || render_minor > 255U) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid VM acquisition identity");
  }
  request_owner(&request, handle);
  request.argument_words[0] = gpu_id;
  request.argument_words[1] = render_minor;
  return exchange(provider, &request, options, &result, error, error_size);
}

sagr_kmt_status_t sagr_kmt_set_memory_policy(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t gpu_id, uint32_t default_policy, uint32_t alternate_policy,
    uint32_t misc_process_flags, uint64_t alternate_aperture_base,
    uint64_t alternate_aperture_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_SET_MEMORY_POLICY, options, &request, error,
      error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (gpu_id == 0U ||
      (default_policy != SAGR_KMT_CACHE_POLICY_COHERENT &&
       default_policy != SAGR_KMT_CACHE_POLICY_NONCOHERENT) ||
      (alternate_policy != SAGR_KMT_CACHE_POLICY_COHERENT &&
       alternate_policy != SAGR_KMT_CACHE_POLICY_NONCOHERENT) ||
      alternate_aperture_size == 0U ||
      alternate_aperture_base > UINT64_MAX - alternate_aperture_size ||
      (alternate_aperture_base & UINT64_C(0xffff)) != 0U ||
      (alternate_aperture_size & UINT64_C(0xffff)) != 0U) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid memory policy carriers");
  }
  request_owner(&request, handle);
  request.argument_words[0] = gpu_id;
  request.argument_words[1] = default_policy;
  request.argument_words[2] = alternate_policy;
  request.argument_words[3] = misc_process_flags;
  put_word_u64(request.argument_words, 4, alternate_aperture_base);
  put_word_u64(request.argument_words, 6, alternate_aperture_size);
  return exchange(provider, &request, options, &result, error, error_size);
}

sagr_kmt_status_t sagr_kmt_alloc_memory(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_alloc_options_t *alloc, sagr_kmt_handle_t *out_memory,
    sagr_kmt_memory_info_t *out_info, uint32_t info_size,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_handle_t committed_handle;
  sagr_kmt_memory_info_t committed_info;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_ALLOC_MEMORY, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (alloc == NULL || alloc->struct_size < sizeof(*alloc) ||
      alloc->flags != 0 || alloc->node_id != 1U || alloc->reserved0 != 0 ||
      alloc->size_bytes == 0 ||
      (alloc->alignment_bytes != SAGR_MEMORY_ALIGNMENT_4K &&
       alloc->alignment_bytes != SAGR_MEMORY_ALIGNMENT_64K) ||
      !bytes_are_zero(alloc->reserved, sizeof(alloc->reserved)) ||
      out_memory == NULL || out_info == NULL || info_size < sizeof(*out_info)) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid memory allocation carriers");
  }
  request_owner(&request, handle);
  request.argument_words[0] = alloc->node_id;
  put_word_u64(request.argument_words, 1, alloc->size_bytes);
  put_word_u64(request.argument_words, 3, alloc->alignment_bytes);
  put_word_u64(request.argument_words, 5, alloc->memory_flags);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  committed_handle.owner_id = result.owner_id;
  committed_handle.owner_generation = result.owner_generation;
  committed_handle.object_id = result.object_id;
  committed_handle.object_generation = result.object_generation;
  memset(&committed_info, 0, sizeof(committed_info));
  committed_info.struct_size = (uint32_t)sizeof(committed_info);
  committed_info.flags = result.result_words[0];
  committed_info.size_bytes = get_word_u64(result.result_words, 1);
  committed_info.alignment_bytes = get_word_u64(result.result_words, 3);
  committed_info.simulated_gpu_va = get_word_u64(result.result_words, 5);
  *out_memory = committed_handle;
  memcpy(out_info, &committed_info, sizeof(committed_info));
  return status;
}

sagr_kmt_status_t sagr_kmt_free_memory(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_FREE_MEMORY, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_resource_handle(handle, memory, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  request_owner(&request, memory);
  request_object(&request, memory);
  return exchange(provider, &request, options, &result, error, error_size);
}

sagr_kmt_status_t sagr_kmt_copy_memory(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const sagr_kmt_copy_options_t *copy,
    const void *source, void *destination, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_COPY_MEMORY, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_resource_handle(handle, memory, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (copy == NULL || copy->struct_size < sizeof(*copy) ||
      (copy->flags != SAGR_KMT_COPY_HOST_TO_SIM &&
       copy->flags != SAGR_KMT_COPY_SIM_TO_HOST) || copy->byte_count == 0 ||
      copy->byte_count > SAGR_KMT_BUFFER_BYTES ||
      copy->offset_bytes > UINT64_MAX - copy->byte_count ||
      !bytes_are_zero(copy->reserved, sizeof(copy->reserved)) ||
      (copy->flags == SAGR_KMT_COPY_HOST_TO_SIM &&
       (source == NULL || destination != NULL)) ||
      (copy->flags == SAGR_KMT_COPY_SIM_TO_HOST &&
       (destination == NULL || source != NULL))) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid memory copy carriers");
  }
  request_owner(&request, memory);
  request_object(&request, memory);
  request.argument_words[0] = copy->flags;
  put_word_u64(request.argument_words, 1, copy->offset_bytes);
  put_word_u64(request.argument_words, 3, copy->byte_count);
  if (copy->flags == SAGR_KMT_COPY_HOST_TO_SIM) {
    request.buffer_bytes = (uint32_t)copy->byte_count;
    memcpy(request.buffer, source, request.buffer_bytes);
    request.buffer_crc32c = sagr_crc32c(request.buffer, request.buffer_bytes);
  }
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (copy->flags == SAGR_KMT_COPY_SIM_TO_HOST) {
    if (result.buffer_bytes != copy->byte_count) {
      return fail_local(error, error_size,
                        SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                        "KMT copy result length mismatch");
    }
    memcpy(destination, result.buffer, result.buffer_bytes);
  } else if (result.buffer_bytes != 0) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "KMT host-to-sim copy returned unexpected bytes");
  }
  return status;
}

sagr_kmt_status_t sagr_kmt_queue_create(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_queue_options_t *queue, sagr_kmt_handle_t *out_queue,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_handle_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_QUEUE_CREATE, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (queue == NULL || queue->struct_size < sizeof(*queue) ||
      queue->flags != 0 || queue->node_id != 1U || queue->depth == 0 ||
      queue->depth > SAGR_KMT_QUEUE_MAX_DEPTH ||
      queue->ring_size_bytes == 0 ||
      queue->ring_base_address == 0 ||
      (queue->ring_base_address & UINT64_C(63)) != 0 ||
      queue->read_pointer_address == 0 ||
      queue->write_pointer_address == 0 ||
      (queue->read_pointer_address & UINT64_C(7)) != 0 ||
      (queue->write_pointer_address & UINT64_C(7)) != 0 ||
      queue->ring_size_bytes != (uint64_t)queue->depth * UINT64_C(64) ||
      queue->reserved0 != 0 ||
      !bytes_are_zero(queue->reserved, sizeof(queue->reserved)) ||
      out_queue == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid queue creation carriers");
  }
  request_owner(&request, handle);
  request.argument_words[0] = queue->node_id;
  request.argument_words[1] = queue->queue_type;
  request.argument_words[2] = queue->depth;
  put_word_u64(request.argument_words, 3, queue->ring_base_address);
  request.buffer_bytes = 24U;
  put_buffer_u64(request.buffer, 0U, queue->ring_size_bytes);
  put_buffer_u64(request.buffer, 8U, queue->read_pointer_address);
  put_buffer_u64(request.buffer, 16U, queue->write_pointer_address);
  request.buffer_crc32c = sagr_crc32c(request.buffer, request.buffer_bytes);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (result.result_words[0] != queue->depth ||
      (get_word_u64(result.result_words, 1) %
       SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_SLOT_BYTES) != 0U ||
      result.result_words[3] != 0U || result.result_words[4] != 0U ||
      result.result_words[5] != 0U || result.result_words[6] != 0U ||
      result.result_words[7] != 0U || result.buffer_bytes != 0U) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "noncanonical queue creation result");
  }
  committed.owner_id = result.owner_id;
  committed.owner_generation = result.owner_generation;
  committed.object_id = result.object_id;
  committed.object_generation = result.object_generation;
  *out_queue = committed;
  return status;
}

sagr_kmt_status_t sagr_kmt_queue_destroy(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *queue, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_QUEUE_DESTROY, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_resource_handle(handle, queue, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  request_owner(&request, queue);
  request_object(&request, queue);
  return exchange(provider, &request, options, &result, error, error_size);
}

sagr_kmt_status_t sagr_kmt_queue_doorbell(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *queue, uint64_t command_kind, uint64_t *sequence,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  uint64_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_QUEUE_DOORBELL, options, &request, error,
      error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_resource_handle(handle, queue, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (sequence == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "doorbell sequence output is null");
  }
  request_owner(&request, queue);
  request_object(&request, queue);
  put_word_u64(request.argument_words, 0, command_kind);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  committed = get_word_u64(result.result_words, 0);
  if (committed == 0) {
    return fail_local(error, error_size,
                      SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                      "doorbell result sequence is zero");
  }
  *sequence = committed;
  return status;
}

static sagr_kmt_status_t event_unary(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, uint16_t operation, int64_t value,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, operation, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_resource_handle(handle, event, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  request_owner(&request, event);
  request_object(&request, event);
  put_word_u64(request.argument_words, 0, (uint64_t)value);
  return exchange(provider, &request, options, &result, error, error_size);
}

sagr_kmt_status_t sagr_kmt_event_create(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_event_options_t *event, sagr_kmt_handle_t *out_event,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_handle_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_EVENT_CREATE, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (event == NULL || event->struct_size < sizeof(*event) ||
      event->flags != 0 ||
      !bytes_are_zero(event->reserved, sizeof(event->reserved)) ||
      out_event == NULL) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid event creation carriers");
  }
  request_owner(&request, handle);
  put_word_u64(request.argument_words, 0, (uint64_t)event->initial_value);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  committed.owner_id = result.owner_id;
  committed.owner_generation = result.owner_generation;
  committed.object_id = result.object_id;
  committed.object_generation = result.object_generation;
  *out_event = committed;
  return status;
}

sagr_kmt_status_t sagr_kmt_event_destroy(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  return event_unary(provider, handle, event, SAGR_KMT_OP_EVENT_DESTROY, 0,
                     options, error, error_size);
}

sagr_kmt_status_t sagr_kmt_event_set(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, int64_t value,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  return event_unary(provider, handle, event, SAGR_KMT_OP_EVENT_SET, value,
                     options, error, error_size);
}

sagr_kmt_status_t sagr_kmt_event_reset(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  return event_unary(provider, handle, event, SAGR_KMT_OP_EVENT_RESET, 0,
                     options, error, error_size);
}

static sagr_kmt_status_t event_result_operation(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, uint16_t operation, uint64_t condition,
    int64_t compare_value, sagr_kmt_event_result_t *out_result,
    uint32_t result_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_event_result_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, operation, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_resource_handle(handle, event, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (out_result == NULL || result_size < sizeof(*out_result) ||
      (operation == SAGR_KMT_OP_EVENT_WAIT &&
       condition > SAGR_SIGNAL_CONDITION_GTE)) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid event result carriers");
  }
  request_owner(&request, event);
  request_object(&request, event);
  put_word_u64(request.argument_words, 0, condition);
  put_word_u64(request.argument_words, 2, (uint64_t)compare_value);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  memset(&committed, 0, sizeof(committed));
  committed.struct_size = (uint32_t)sizeof(committed);
  committed.value = (int64_t)get_word_u64(result.result_words, 0);
  if (operation == SAGR_KMT_OP_EVENT_QUERY) {
    committed.ready = result.result_words[2];
    committed.sequence = get_word_u64(result.result_words, 3);
  }
  memcpy(out_result, &committed, sizeof(committed));
  return status;
}

sagr_kmt_status_t sagr_kmt_event_query(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, sagr_kmt_event_result_t *out_result,
    uint32_t result_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  return event_result_operation(provider, handle, event,
                                SAGR_KMT_OP_EVENT_QUERY, 0, 0, out_result,
                                result_size, options, error, error_size);
}

sagr_kmt_status_t sagr_kmt_event_wait(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, uint64_t condition, int64_t compare_value,
    sagr_kmt_event_result_t *out_result, uint32_t result_size,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size) {
  return event_result_operation(provider, handle, event,
                                SAGR_KMT_OP_EVENT_WAIT, condition,
                                compare_value, out_result, result_size, options,
                                error, error_size);
}

sagr_kmt_status_t sagr_kmt_pointer_info(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, sagr_kmt_pointer_info_t *out_info,
    uint32_t info_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_pointer_info_t committed;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_POINTER_INFO, options, &request, error, error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_resource_handle(handle, memory, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (out_info == NULL || info_size < sizeof(*out_info)) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "pointer-info output buffer is invalid");
  }
  request_owner(&request, memory);
  request_object(&request, memory);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  memset(&committed, 0, sizeof(committed));
  committed.struct_size = (uint32_t)sizeof(committed);
  committed.flags = result.result_words[0];
  committed.allocation = *memory;
  committed.offset_bytes = get_word_u64(result.result_words, 1);
  committed.size_bytes = get_word_u64(result.result_words, 3);
  memcpy(out_info, &committed, sizeof(committed));
  return status;
}

sagr_kmt_status_t sagr_kmt_model_drm_call(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_model_drm_call_t *call, void *result_buffer,
    uint32_t result_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size) {
  static const uint32_t command_sizes[SAGR_PROVIDER_MODEL_DRM_COMMAND_COUNT] = {
      48U, 8U, 16U, 24U, 16U, 16U, 16U, 32U,
      16U, 4U, 32U, 8U, 16U, 16U, 16U};
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_status_t status = prepare_request(
      provider, SAGR_KMT_OP_MODEL_DRM_CALL, options, &request, error,
      error_size);
  if (status == SAGR_KMT_STATUS_SUCCESS) {
    status = validate_kfd_handle(handle, error, error_size);
  }
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (call == NULL || call->struct_size < sizeof(*call) || call->flags != 0 ||
      call->command >= SAGR_PROVIDER_MODEL_DRM_COMMAND_COUNT ||
      call->argument_bytes != command_sizes[call->command] ||
      call->argument_bytes > SAGR_KMT_BUFFER_BYTES ||
      (result_size != 0U && result_buffer == NULL) ||
      result_size > SAGR_KMT_BUFFER_BYTES) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_INVALID_PARAMETER,
                      "invalid model DRM call carriers");
  }
  request_owner(&request, handle);
  request.argument_words[0] = call->command;
  request.argument_words[1] = result_size;
  request.buffer_bytes = call->argument_bytes;
  memcpy(request.buffer, call->argument, call->argument_bytes);
  request.buffer_crc32c = sagr_crc32c(request.buffer, request.buffer_bytes);
  status = exchange(provider, &request, options, &result, error, error_size);
  if (status != SAGR_KMT_STATUS_SUCCESS) {
    return status;
  }
  if (result.buffer_bytes > result_size) {
    return fail_local(error, error_size, SAGR_KMT_STATUS_BUFFER_TOO_SMALL,
                      "model DRM result buffer is too small");
  }
  if (result.buffer_bytes != 0) {
    memcpy(result_buffer, result.buffer, result.buffer_bytes);
  }
  return status;
}

_Static_assert(sizeof(sagr_kmt_envelope_request_t) == SAGR_KMT_PAYLOAD_BYTES,
               "KMT request payload ABI changed");
_Static_assert(sizeof(sagr_kmt_envelope_result_t) == SAGR_KMT_PAYLOAD_BYTES,
               "KMT result payload ABI changed");
_Static_assert(offsetof(sagr_kmt_envelope_request_t, owner_generation) == 24,
               "KMT request owner generation offset changed");
_Static_assert(offsetof(sagr_kmt_envelope_request_t, buffer) == 104,
               "KMT request copied-buffer offset changed");
_Static_assert(offsetof(sagr_kmt_envelope_result_t, owner_generation) == 32,
               "KMT result owner generation offset changed");
_Static_assert(offsetof(sagr_kmt_envelope_result_t, buffer) == 112,
               "KMT result copied-buffer offset changed");
