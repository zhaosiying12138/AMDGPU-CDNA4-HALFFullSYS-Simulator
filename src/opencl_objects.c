/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "opencl_internal.h"

#include <stdlib.h>
#include <string.h>

struct _cl_platform_id sagr_cl_platform = {SAGR_CL_PLATFORM_MAGIC};
struct _cl_device_id sagr_cl_device = {SAGR_CL_DEVICE_MAGIC,
                                       &sagr_cl_platform};

int sagr_cl_valid_platform(cl_platform_id platform) {
  return platform == &sagr_cl_platform &&
         platform->magic == SAGR_CL_PLATFORM_MAGIC;
}

int sagr_cl_valid_device(cl_device_id device) {
  return device == &sagr_cl_device && device->magic == SAGR_CL_DEVICE_MAGIC;
}

int sagr_cl_valid_context(cl_context context) {
  return context != NULL && context->magic == SAGR_CL_CONTEXT_MAGIC;
}

int sagr_cl_valid_queue(cl_command_queue queue) {
  return queue != NULL && queue->magic == SAGR_CL_QUEUE_MAGIC &&
         sagr_cl_valid_context(queue->context);
}

int sagr_cl_valid_program(cl_program program) {
  return program != NULL && program->magic == SAGR_CL_PROGRAM_MAGIC &&
         sagr_cl_valid_context(program->context);
}

int sagr_cl_valid_kernel(cl_kernel kernel) {
  return kernel != NULL && kernel->magic == SAGR_CL_KERNEL_MAGIC &&
         sagr_cl_valid_program(kernel->program);
}

int sagr_cl_valid_memory(cl_mem memory) {
  return memory != NULL && memory->magic == SAGR_CL_MEMORY_MAGIC &&
         sagr_cl_valid_context(memory->context);
}

cl_int sagr_cl_copy_info(const void *value, size_t value_size,
                         size_t parameter_size, void *parameter_value,
                         size_t *parameter_size_ret) {
  if (parameter_value != NULL && parameter_size < value_size) {
    return CL_INVALID_VALUE;
  }
  if (parameter_value != NULL && value_size != 0U) {
    memcpy(parameter_value, value, value_size);
  }
  if (parameter_size_ret != NULL) {
    *parameter_size_ret = value_size;
  }
  return CL_SUCCESS;
}

cl_int sagr_cl_copy_string(const char *value, size_t parameter_size,
                           void *parameter_value,
                           size_t *parameter_size_ret) {
  return value == NULL
             ? CL_INVALID_VALUE
             : sagr_cl_copy_info(value, strlen(value) + 1U, parameter_size,
                                 parameter_value, parameter_size_ret);
}

cl_int sagr_cl_status_to_error(sagr_status_t status) {
  switch (status) {
    case SAGR_STATUS_SUCCESS:
      return CL_SUCCESS;
    case SAGR_STATUS_INVALID_ARGUMENT:
    case SAGR_STATUS_INVALID_HANDLE:
    case SAGR_STATUS_BUFFER_TOO_SMALL:
      return CL_INVALID_VALUE;
    case SAGR_STATUS_NOT_SUPPORTED:
    case SAGR_STATUS_CAPABILITY_MISMATCH:
    case SAGR_STATUS_VERSION_MISMATCH:
      return CL_INVALID_OPERATION;
    case SAGR_STATUS_OUT_OF_RESOURCES:
      return CL_OUT_OF_HOST_MEMORY;
    case SAGR_STATUS_TIMED_OUT:
    case SAGR_STATUS_UNAVAILABLE:
    case SAGR_STATUS_CONNECTION_LOST:
    case SAGR_STATUS_ENDPOINT_NOT_FOUND:
    case SAGR_STATUS_BUSY:
    case SAGR_STATUS_CANCELLED:
      return CL_OUT_OF_RESOURCES;
    default:
      return CL_OUT_OF_RESOURCES;
  }
}

void sagr_cl_context_retain_internal(cl_context context) {
  (void)atomic_fetch_add_explicit(&context->references, 1U,
                                  memory_order_relaxed);
}

void sagr_cl_program_retain_internal(cl_program program) {
  (void)atomic_fetch_add_explicit(&program->references, 1U,
                                  memory_order_relaxed);
}

void sagr_cl_memory_retain_internal(cl_mem memory) {
  (void)atomic_fetch_add_explicit(&memory->references, 1U,
                                  memory_order_relaxed);
}

void sagr_cl_context_release_internal(cl_context context) {
  if (atomic_fetch_sub_explicit(&context->references, 1U,
                                memory_order_acq_rel) != 1U) {
    return;
  }
  (void)pthread_mutex_lock(&context->mutex);
  sagr_cl_simulator_shutdown(&context->simulator);
  (void)pthread_mutex_unlock(&context->mutex);
  (void)pthread_mutex_destroy(&context->mutex);
  context->magic = 0U;
  free(context);
}

void sagr_cl_program_release_internal(cl_program program) {
  cl_context context;
  if (atomic_fetch_sub_explicit(&program->references, 1U,
                                memory_order_acq_rel) != 1U) {
    return;
  }
  context = program->context;
  program->magic = 0U;
  free(program->source);
  free(program->image);
  free(program->build_options);
  free(program->build_log);
  free(program);
  sagr_cl_context_release_internal(context);
}

void sagr_cl_memory_release_internal(cl_mem memory) {
  cl_context context;
  if (atomic_fetch_sub_explicit(&memory->references, 1U,
                                memory_order_acq_rel) != 1U) {
    return;
  }
  context = memory->context;
  if (memory->memory != NULL) {
    (void)pthread_mutex_lock(&context->mutex);
    (void)sagr_memory_free(&memory->memory, NULL, NULL, 0U);
    (void)pthread_mutex_unlock(&context->mutex);
  }
  memory->magic = 0U;
  free(memory->initial_data);
  free(memory);
  sagr_cl_context_release_internal(context);
}

SAGR_CL_EXPORT cl_int CL_API_CALL clRetainContext(cl_context context) {
  if (!sagr_cl_valid_context(context)) {
    return CL_INVALID_CONTEXT;
  }
  sagr_cl_context_retain_internal(context);
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clReleaseContext(cl_context context) {
  if (!sagr_cl_valid_context(context)) {
    return CL_INVALID_CONTEXT;
  }
  sagr_cl_context_release_internal(context);
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_int CL_API_CALL
clRetainCommandQueue(cl_command_queue command_queue) {
  if (!sagr_cl_valid_queue(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  (void)atomic_fetch_add_explicit(&command_queue->references, 1U,
                                  memory_order_relaxed);
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue command_queue) {
  cl_context context;
  cl_int result = CL_SUCCESS;
  if (!sagr_cl_valid_queue(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  if (atomic_fetch_sub_explicit(&command_queue->references, 1U,
                                memory_order_acq_rel) != 1U) {
    return CL_SUCCESS;
  }
  context = command_queue->context;
  if (command_queue->queue != NULL) {
    sagr_error_info_t error;
    sagr_status_t status;
    memset(&error, 0, sizeof(error));
    (void)pthread_mutex_lock(&context->mutex);
    status = sagr_queue_destroy(&command_queue->queue, NULL, &error,
                                (uint32_t)sizeof(error));
    (void)pthread_mutex_unlock(&context->mutex);
    result = sagr_cl_status_to_error(status);
  }
  command_queue->magic = 0U;
  free(command_queue);
  sagr_cl_context_release_internal(context);
  return result;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clRetainProgram(cl_program program) {
  if (!sagr_cl_valid_program(program)) {
    return CL_INVALID_PROGRAM;
  }
  sagr_cl_program_retain_internal(program);
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clReleaseProgram(cl_program program) {
  if (!sagr_cl_valid_program(program)) {
    return CL_INVALID_PROGRAM;
  }
  sagr_cl_program_release_internal(program);
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clRetainKernel(cl_kernel kernel) {
  if (!sagr_cl_valid_kernel(kernel)) {
    return CL_INVALID_KERNEL;
  }
  (void)atomic_fetch_add_explicit(&kernel->references, 1U,
                                  memory_order_relaxed);
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) {
  cl_program program;
  uint32_t index;
  if (!sagr_cl_valid_kernel(kernel)) {
    return CL_INVALID_KERNEL;
  }
  if (atomic_fetch_sub_explicit(&kernel->references, 1U,
                                memory_order_acq_rel) != 1U) {
    return CL_SUCCESS;
  }
  program = kernel->program;
  kernel->magic = 0U;
  for (index = 0U; index < SAGR_CODE_OBJECT_MAX_ARGS; ++index) {
    if (kernel->args[index].is_memory && kernel->args[index].memory != NULL) {
      sagr_cl_memory_release_internal(kernel->args[index].memory);
    }
  }
  free(kernel);
  sagr_cl_program_release_internal(program);
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clRetainMemObject(cl_mem memobj) {
  if (!sagr_cl_valid_memory(memobj)) {
    return CL_INVALID_MEM_OBJECT;
  }
  sagr_cl_memory_retain_internal(memobj);
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clReleaseMemObject(cl_mem memobj) {
  if (!sagr_cl_valid_memory(memobj)) {
    return CL_INVALID_MEM_OBJECT;
  }
  sagr_cl_memory_release_internal(memobj);
  return CL_SUCCESS;
}
