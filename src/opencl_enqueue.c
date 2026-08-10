/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "opencl_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void queue_operation(sagr_queue_operation_options_t *options) {
  (void)sagr_queue_operation_options_init(options,
                                          (uint32_t)sizeof(*options));
  options->timeout_ns = SAGR_CL_PROCESS_TIMEOUT_MS * UINT64_C(1000000);
}

static void memory_operation(sagr_memory_operation_options_t *options) {
  (void)sagr_memory_operation_options_init(options,
                                           (uint32_t)sizeof(*options));
  options->timeout_ns = SAGR_CL_PROCESS_TIMEOUT_MS * UINT64_C(1000000);
}

static cl_int copy_to_device(cl_mem memory, uint64_t offset,
                             const void *source, uint64_t size) {
  const uint8_t *cursor = (const uint8_t *)source;
  sagr_memory_operation_options_t operation;
  sagr_error_info_t error;
  while (size != 0U) {
    const uint64_t chunk =
        size < SAGR_MEMORY_MAX_TRANSFER_BYTES ? size
                                              : SAGR_MEMORY_MAX_TRANSFER_BYTES;
    sagr_status_t status;
    memory_operation(&operation);
    memset(&error, 0, sizeof(error));
    status = sagr_memory_copy_from_host(memory->memory, offset, cursor, chunk,
                                        &operation, &error,
                                        (uint32_t)sizeof(error));
    if (status != SAGR_STATUS_SUCCESS) {
      return sagr_cl_status_to_error(status);
    }
    offset += chunk;
    cursor += (size_t)chunk;
    size -= chunk;
  }
  return CL_SUCCESS;
}

static cl_int copy_from_device(cl_mem memory, uint64_t offset,
                               void *destination, uint64_t size) {
  uint8_t *cursor = (uint8_t *)destination;
  sagr_memory_operation_options_t operation;
  sagr_error_info_t error;
  while (size != 0U) {
    const uint64_t chunk =
        size < SAGR_MEMORY_MAX_TRANSFER_BYTES ? size
                                              : SAGR_MEMORY_MAX_TRANSFER_BYTES;
    sagr_status_t status;
    memory_operation(&operation);
    memset(&error, 0, sizeof(error));
    status = sagr_memory_copy_to_host(memory->memory, offset, cursor, chunk,
                                      &operation, &error,
                                      (uint32_t)sizeof(error));
    if (status != SAGR_STATUS_SUCCESS) {
      return sagr_cl_status_to_error(status);
    }
    offset += chunk;
    cursor += (size_t)chunk;
    size -= chunk;
  }
  return CL_SUCCESS;
}

cl_int sagr_cl_ensure_queue(cl_command_queue queue) {
  sagr_queue_create_options_t options;
  sagr_queue_operation_options_t operation;
  sagr_error_info_t error;
  sagr_status_t status;
  cl_int result;
  if (queue->queue != NULL) {
    return CL_SUCCESS;
  }
  result = sagr_cl_simulator_ensure(&queue->context->simulator);
  if (result != CL_SUCCESS) {
    return result;
  }
  if (sagr_queue_create_options_init(&options, (uint32_t)sizeof(options)) !=
      SAGR_STATUS_SUCCESS) {
    return CL_OUT_OF_RESOURCES;
  }
  options.depth = 1U;
  queue_operation(&operation);
  memset(&queue->queue_info, 0, sizeof(queue->queue_info));
  memset(&error, 0, sizeof(error));
  status = sagr_queue_create(queue->context->simulator.instance, &options,
                             &operation, &queue->queue, &queue->queue_info,
                             (uint32_t)sizeof(queue->queue_info), &error,
                             (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS && error.message[0] != '\0') {
    (void)fprintf(stderr,
                  "self-amdgpu OpenCL queue creation failed: runtime=%d "
                  "wire=%d errno=%d: %s\n",
                  (int)status, (int)error.wire_status,
                  (int)error.native_errno, error.message);
  }
  return sagr_cl_status_to_error(status);
}

cl_int sagr_cl_ensure_memory(cl_mem memory) {
  sagr_memory_allocate_options_t options;
  sagr_memory_operation_options_t operation;
  sagr_error_info_t error;
  sagr_status_t status;
  cl_int result;
  if (memory->memory != NULL) {
    return CL_SUCCESS;
  }
  result = sagr_cl_simulator_ensure(&memory->context->simulator);
  if (result != CL_SUCCESS) {
    return result;
  }
  if (sagr_memory_allocate_options_init(&options, (uint32_t)sizeof(options)) !=
      SAGR_STATUS_SUCCESS) {
    return CL_OUT_OF_RESOURCES;
  }
  options.size_bytes = (uint64_t)memory->size;
  options.alignment_bytes = SAGR_MEMORY_ALIGNMENT_4K;
  memory_operation(&operation);
  memset(&memory->memory_info, 0, sizeof(memory->memory_info));
  memset(&error, 0, sizeof(error));
  status = sagr_memory_allocate(memory->context->simulator.instance, &options,
                                &operation, &memory->memory,
                                &memory->memory_info,
                                (uint32_t)sizeof(memory->memory_info), &error,
                                (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS) {
    if (error.message[0] != '\0') {
      (void)fprintf(stderr,
                    "self-amdgpu OpenCL memory allocation failed: runtime=%d "
                    "wire=%d errno=%d: %s\n",
                    (int)status, (int)error.wire_status,
                    (int)error.native_errno, error.message);
    }
    return sagr_cl_status_to_error(status);
  }
  if (memory->initial_data != NULL) {
    result = copy_to_device(memory, 0U, memory->initial_data,
                            (uint64_t)memory->size);
    if (result != CL_SUCCESS) {
      (void)sagr_memory_free(&memory->memory, &operation, NULL, 0U);
      memset(&memory->memory_info, 0, sizeof(memory->memory_info));
      return result;
    }
    free(memory->initial_data);
    memory->initial_data = NULL;
  }
  return CL_SUCCESS;
}

static cl_int validate_transfer(cl_command_queue queue, cl_mem buffer,
                                cl_bool blocking, size_t offset, size_t size,
                                const void *pointer, cl_uint wait_count,
                                const cl_event *wait_list, cl_event *event) {
  if (!sagr_cl_valid_queue(queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  if (!sagr_cl_valid_memory(buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }
  if (buffer->context != queue->context) {
    return CL_INVALID_CONTEXT;
  }
  if (blocking != CL_TRUE) {
    return CL_INVALID_OPERATION;
  }
  if (pointer == NULL || size == 0U || offset > buffer->size ||
      size > buffer->size - offset) {
    return CL_INVALID_VALUE;
  }
  if ((wait_count == 0U) != (wait_list == NULL)) {
    return CL_INVALID_EVENT_WAIT_LIST;
  }
  if (wait_count != 0U || event != NULL) {
    return CL_INVALID_OPERATION;
  }
  return CL_SUCCESS;
}

cl_int sagr_cl_enqueue_write(cl_command_queue queue, cl_mem buffer,
                             cl_bool blocking_write, size_t offset, size_t size,
                             const void *pointer, cl_uint wait_count,
                             const cl_event *wait_list, cl_event *event) {
  cl_int result = validate_transfer(queue, buffer, blocking_write, offset, size,
                                    pointer, wait_count, wait_list, event);
  if (result != CL_SUCCESS) {
    return result;
  }
  (void)pthread_mutex_lock(&queue->context->mutex);
  result = sagr_cl_ensure_memory(buffer);
  if (result == CL_SUCCESS) {
    result = copy_to_device(buffer, (uint64_t)offset, pointer, (uint64_t)size);
  }
  (void)pthread_mutex_unlock(&queue->context->mutex);
  return result;
}

cl_int sagr_cl_enqueue_read(cl_command_queue queue, cl_mem buffer,
                            cl_bool blocking_read, size_t offset, size_t size,
                            void *pointer, cl_uint wait_count,
                            const cl_event *wait_list, cl_event *event) {
  cl_int result = validate_transfer(queue, buffer, blocking_read, offset, size,
                                    pointer, wait_count, wait_list, event);
  if (result != CL_SUCCESS) {
    return result;
  }
  (void)pthread_mutex_lock(&queue->context->mutex);
  result = sagr_cl_ensure_memory(buffer);
  if (result == CL_SUCCESS) {
    result = copy_from_device(buffer, (uint64_t)offset, pointer,
                              (uint64_t)size);
  }
  (void)pthread_mutex_unlock(&queue->context->mutex);
  return result;
}

static const sagr_code_object_arg_info_t *visible_metadata(
    const sagr_code_object_kernel_info_t *kernel, uint32_t requested) {
  uint32_t index;
  uint32_t visible = 0U;
  for (index = 0U; index < kernel->arg_count; ++index) {
    if (kernel->args[index].kind != SAGR_CODE_OBJECT_ARG_VISIBLE) {
      continue;
    }
    if (visible == requested) {
      return &kernel->args[index];
    }
    ++visible;
  }
  return NULL;
}

static uint64_t scalar_value(const uint8_t *bytes, size_t size) {
  uint64_t result = 0U;
  size_t index;
  for (index = 0U; index < size; ++index) {
    result |= (uint64_t)bytes[index] << (index * 8U);
  }
  return result;
}

static int hidden_value(const char *kind, const uint64_t global[3],
                        const uint64_t local[3], const uint64_t offset[3],
                        uint32_t dimensions, uint64_t *value) {
  if (strcmp(kind, "hidden_block_count_x") == 0) {
    *value = global[0] / local[0];
  } else if (strcmp(kind, "hidden_block_count_y") == 0) {
    *value = global[1] / local[1];
  } else if (strcmp(kind, "hidden_block_count_z") == 0) {
    *value = global[2] / local[2];
  } else if (strcmp(kind, "hidden_group_size_x") == 0) {
    *value = local[0];
  } else if (strcmp(kind, "hidden_group_size_y") == 0) {
    *value = local[1];
  } else if (strcmp(kind, "hidden_group_size_z") == 0) {
    *value = local[2];
  } else if (strcmp(kind, "hidden_remainder_x") == 0) {
    *value = global[0] % local[0];
  } else if (strcmp(kind, "hidden_remainder_y") == 0) {
    *value = global[1] % local[1];
  } else if (strcmp(kind, "hidden_remainder_z") == 0) {
    *value = global[2] % local[2];
  } else if (strcmp(kind, "hidden_global_offset_x") == 0) {
    *value = offset[0];
  } else if (strcmp(kind, "hidden_global_offset_y") == 0) {
    *value = offset[1];
  } else if (strcmp(kind, "hidden_global_offset_z") == 0) {
    *value = offset[2];
  } else if (strcmp(kind, "hidden_grid_dims") == 0) {
    *value = dimensions;
  } else if (strcmp(kind, "hidden_none") == 0) {
    *value = 0U;
  } else {
    return 0;
  }
  return 1;
}

static cl_int materialize_kernarg(
    cl_kernel kernel, const uint64_t global[3], const uint64_t local[3],
    const uint64_t offset[3], uint32_t dimensions, uint8_t *bytes) {
  sagr_code_object_arg_value_t values[SAGR_CODE_OBJECT_MAX_ARGS];
  uint32_t value_count = 0U;
  uint32_t visible = 0U;
  uint32_t index;
  uint32_t written = 0U;
  for (index = 0U; index < kernel->info.arg_count; ++index) {
    const sagr_code_object_arg_info_t *metadata = &kernel->info.args[index];
    uint64_t value = 0U;
    if (metadata->size_bytes > sizeof(value)) {
      return CL_INVALID_ARG_SIZE;
    }
    if (metadata->kind == SAGR_CODE_OBJECT_ARG_VISIBLE) {
      const struct sagr_cl_kernel_arg *slot;
      const sagr_code_object_arg_info_t *visible_info =
          visible_metadata(&kernel->info, visible);
      if (visible >= SAGR_CODE_OBJECT_MAX_ARGS || visible_info != metadata) {
        return CL_INVALID_KERNEL_DEFINITION;
      }
      slot = &kernel->args[visible++];
      if (!slot->is_set) {
        return CL_INVALID_KERNEL_ARGS;
      }
      if (slot->is_memory) {
        cl_int result = sagr_cl_ensure_memory(slot->memory);
        if (result != CL_SUCCESS) {
          return result;
        }
        value = slot->memory->memory_info.simulated_va;
      } else {
        value = scalar_value(slot->bytes, slot->size);
      }
    } else if (metadata->kind == SAGR_CODE_OBJECT_ARG_HIDDEN) {
      if (!hidden_value(metadata->value_kind, global, local, offset, dimensions,
                        &value)) {
        return CL_INVALID_KERNEL_DEFINITION;
      }
    } else {
      return CL_INVALID_KERNEL_DEFINITION;
    }
    memset(&values[value_count], 0, sizeof(values[value_count]));
    values[value_count].struct_size = (uint32_t)sizeof(values[value_count]);
    values[value_count].arg_index = metadata->index;
    values[value_count].value = value;
    ++value_count;
  }
  if (visible != kernel->info.visible_arg_count ||
      sagr_code_object_pack_kernarg(
          &kernel->info, values, value_count, bytes,
          kernel->info.kernarg_segment_size, &written) != SAGR_STATUS_SUCCESS ||
      written != kernel->info.kernarg_segment_size) {
    return CL_INVALID_KERNEL_DEFINITION;
  }
  return CL_SUCCESS;
}

static cl_int dispatch_locked(cl_command_queue queue, cl_kernel kernel,
                              const uint64_t global[3],
                              const uint64_t local[3],
                              const uint64_t offset[3], uint32_t dimensions) {
  cl_program program = kernel->program;
  sagr_code_object_remote_info_t remote;
  sagr_generic_map_options_t map_options;
  sagr_generic_mapping_t mapping = NULL;
  sagr_generic_mapping_info_t mapping_info;
  sagr_generic_kernarg_allocate_options_t allocate_options;
  sagr_generic_kernarg_t kernarg = NULL;
  sagr_generic_kernarg_info_t kernarg_info;
  sagr_signal_create_options_t signal_options;
  sagr_signal_t signal = NULL;
  sagr_signal_info_t signal_info;
  sagr_generic_submit_options_t submit_options;
  sagr_generic_dispatch_ticket_t ticket;
  sagr_generic_dispatch_completion_t completion;
  sagr_queue_operation_options_t operation;
  sagr_error_info_t error;
  uint8_t *kernarg_bytes = NULL;
  sagr_status_t status;
  cl_int result;
  int submitted = 0;

  memset(&error, 0, sizeof(error));
  if (kernel->info.kernarg_segment_size == 0U ||
      kernel->info.kernarg_segment_size >
          SAGR_CL_KERNARG_ALLOCATION_BYTES - SAGR_CL_KERNARG_OFFSET ||
      kernel->info.descriptor_kernarg_preload != 0U) {
    return CL_INVALID_PROGRAM_EXECUTABLE;
  }
  result = sagr_cl_ensure_queue(queue);
  if (result != CL_SUCCESS) {
    return result;
  }
  kernarg_bytes = (uint8_t *)malloc(kernel->info.kernarg_segment_size);
  if (kernarg_bytes == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }
  result = materialize_kernarg(kernel, global, local, offset, dimensions,
                               kernarg_bytes);
  if (result != CL_SUCCESS) {
    goto cleanup;
  }
  queue_operation(&operation);
  memset(&remote, 0, sizeof(remote));
  memset(&error, 0, sizeof(error));
  status = sagr_code_object_upload(
      queue->context->simulator.instance, program->image, program->image_size,
      kernel->info.name, &operation, &remote, (uint32_t)sizeof(remote), &error,
      (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS || remote.object_id == 0U ||
      remote.generation == 0U ||
      (program->image_sha256_valid &&
       memcmp(remote.image_sha256, program->image_sha256,
              sizeof(remote.image_sha256)) != 0)) {
    result = status == SAGR_STATUS_SUCCESS ? CL_INVALID_PROGRAM_EXECUTABLE
                                           : sagr_cl_status_to_error(status);
    goto cleanup;
  }
  memcpy(program->image_sha256, remote.image_sha256,
         sizeof(program->image_sha256));
  program->image_sha256_valid = 1;

  (void)sagr_generic_map_options_init(&map_options,
                                      (uint32_t)sizeof(map_options));
  map_options.object_id = remote.object_id;
  map_options.object_generation = remote.generation;
  map_options.kernel_index = kernel->info.index;
  map_options.gfx_target = program->code_object.gfx_target;
  map_options.relocation_count = kernel->info.relocation_count;
  map_options.kernarg_segment_size = kernel->info.kernarg_segment_size;
  map_options.kernarg_segment_align = kernel->info.kernarg_segment_align;
  map_options.descriptor_preload_dwords =
      kernel->info.descriptor_kernarg_preload;
  memcpy(map_options.image_sha256, remote.image_sha256,
         sizeof(map_options.image_sha256));
  if (strlen(kernel->info.name) >= sizeof(map_options.kernel_name)) {
    result = CL_INVALID_KERNEL_NAME;
    goto cleanup;
  }
  memcpy(map_options.kernel_name, kernel->info.name,
         strlen(kernel->info.name) + 1U);
  memset(&mapping_info, 0, sizeof(mapping_info));
  status = sagr_generic_map_object(
      queue->context->simulator.instance, &map_options, &operation, &mapping,
      &mapping_info, (uint32_t)sizeof(mapping_info), &error,
      (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS) {
    result = sagr_cl_status_to_error(status);
    goto cleanup;
  }

  (void)sagr_generic_kernarg_allocate_options_init(
      &allocate_options, (uint32_t)sizeof(allocate_options));
  allocate_options.size_bytes = SAGR_CL_KERNARG_ALLOCATION_BYTES;
  allocate_options.alignment_bytes = 8U;
  memset(&kernarg_info, 0, sizeof(kernarg_info));
  status = sagr_generic_alloc_kernarg(
      mapping, &allocate_options, &operation, &kernarg, &kernarg_info,
      (uint32_t)sizeof(kernarg_info), &error, (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS) {
    result = sagr_cl_status_to_error(status);
    goto cleanup;
  }
  status = sagr_generic_kernarg_copy_from_host(
      kernarg, SAGR_CL_KERNARG_OFFSET, kernarg_bytes,
      kernel->info.kernarg_segment_size, &operation, &error,
      (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS) {
    result = sagr_cl_status_to_error(status);
    goto cleanup;
  }

  (void)sagr_signal_create_options_init(&signal_options,
                                        (uint32_t)sizeof(signal_options));
  signal_options.initial_value = 1;
  memset(&signal_info, 0, sizeof(signal_info));
  status = sagr_signal_create(
      queue->context->simulator.instance, &signal_options, NULL, &signal,
      &signal_info, (uint32_t)sizeof(signal_info), &error,
      (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS) {
    result = sagr_cl_status_to_error(status);
    goto cleanup;
  }

  (void)sagr_generic_submit_options_init(&submit_options,
                                         (uint32_t)sizeof(submit_options));
  submit_options.kernarg_offset = SAGR_CL_KERNARG_OFFSET;
  submit_options.kernarg_size = kernel->info.kernarg_segment_size;
  submit_options.grid_x = (uint32_t)global[0];
  submit_options.grid_y = (uint32_t)global[1];
  submit_options.grid_z = (uint32_t)global[2];
  submit_options.workgroup_x = (uint32_t)local[0];
  submit_options.workgroup_y = (uint32_t)local[1];
  submit_options.workgroup_z = (uint32_t)local[2];
  submit_options.num_warps =
      ((uint32_t)local[0] + kernel->info.wavefront_size - 1U) /
      kernel->info.wavefront_size;
  submit_options.num_ctas = 1U;
  submit_options.shared_memory_bytes = kernel->info.group_segment_fixed_size;
  submit_options.wavefront_size = kernel->info.wavefront_size;
  memset(&ticket, 0, sizeof(ticket));
  status = sagr_queue_submit_generic_dispatch(
      queue->queue, mapping, kernarg, signal, &submit_options, &operation,
      &ticket, (uint32_t)sizeof(ticket), &error, (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS) {
    result = sagr_cl_status_to_error(status);
    goto cleanup;
  }
  submitted = 1;
  memset(&completion, 0, sizeof(completion));
  status = sagr_queue_wait_generic_dispatch(
      queue->queue, &ticket, &operation, &completion,
      (uint32_t)sizeof(completion), &error, (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS ||
      completion.status != SAGR_STATUS_SUCCESS) {
    result = sagr_cl_status_to_error(status != SAGR_STATUS_SUCCESS
                                         ? status
                                         : completion.status);
    goto cleanup;
  }
  result = CL_SUCCESS;

cleanup:
  if (result != CL_SUCCESS && error.message[0] != '\0') {
    (void)fprintf(
        stderr,
        "self-amdgpu OpenCL dispatch failed: cl=%d runtime=%d wire=%d "
        "errno=%d: %s\n",
        result, (int)error.status, (int)error.wire_status,
        (int)error.native_errno, error.message);
  }
  if (mapping != NULL && (!submitted || result == CL_SUCCESS)) {
    status = sagr_generic_unmap_object(&mapping, &operation, &error,
                                       (uint32_t)sizeof(error));
    if (result == CL_SUCCESS && status != SAGR_STATUS_SUCCESS) {
      result = sagr_cl_status_to_error(status);
    }
  }
  if (signal != NULL && (!submitted || result == CL_SUCCESS)) {
    status = sagr_signal_destroy(&signal, NULL, &error,
                                 (uint32_t)sizeof(error));
    if (result == CL_SUCCESS && status != SAGR_STATUS_SUCCESS) {
      result = sagr_cl_status_to_error(status);
    }
  }
  free(kernarg_bytes);
  return result;
}

cl_int sagr_cl_enqueue_kernel(cl_command_queue queue, cl_kernel kernel,
                              cl_uint work_dimensions,
                              const size_t *global_offset,
                              const size_t *global_size,
                              const size_t *local_size, cl_uint wait_count,
                              const cl_event *wait_list, cl_event *event) {
  uint64_t global[3] = {1U, 1U, 1U};
  uint64_t local[3] = {1U, 1U, 1U};
  uint64_t offset[3] = {0U, 0U, 0U};
  cl_int result;
  if (!sagr_cl_valid_queue(queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  if (!sagr_cl_valid_kernel(kernel)) {
    return CL_INVALID_KERNEL;
  }
  if (kernel->program->context != queue->context) {
    return CL_INVALID_CONTEXT;
  }
  if (work_dimensions != 1U || global_size == NULL || local_size == NULL) {
    return CL_INVALID_WORK_DIMENSION;
  }
  if ((wait_count == 0U) != (wait_list == NULL)) {
    return CL_INVALID_EVENT_WAIT_LIST;
  }
  if (wait_count != 0U || event != NULL) {
    return CL_INVALID_OPERATION;
  }
  if (global_size[0] == 0U || global_size[0] > UINT32_MAX) {
    return CL_INVALID_GLOBAL_WORK_SIZE;
  }
  if (local_size[0] == 0U || local_size[0] > 256U ||
      local_size[0] > UINT32_MAX) {
    return CL_INVALID_WORK_GROUP_SIZE;
  }
  if (global_size[0] % local_size[0] != 0U) {
    return CL_INVALID_WORK_GROUP_SIZE;
  }
  if (global_offset != NULL && global_offset[0] != 0U) {
    return CL_INVALID_GLOBAL_OFFSET;
  }
  global[0] = (uint64_t)global_size[0];
  local[0] = (uint64_t)local_size[0];
  (void)pthread_mutex_lock(&queue->context->mutex);
  result = dispatch_locked(queue, kernel, global, local, offset,
                           work_dimensions);
  (void)pthread_mutex_unlock(&queue->context->mutex);
  return result;
}
