/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "opencl_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static void set_error(cl_int *error, cl_int value) {
  if (error != NULL) {
    *error = value;
  }
}

SAGR_CL_EXPORT cl_int CL_API_CALL
clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms,
                 cl_uint *num_platforms) {
  if ((platforms == NULL && num_platforms == NULL) ||
      (platforms != NULL && num_entries == 0U)) {
    return CL_INVALID_VALUE;
  }
  if (platforms != NULL) {
    platforms[0] = &sagr_cl_platform;
  }
  if (num_platforms != NULL) {
    *num_platforms = 1U;
  }
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id platform, cl_platform_info parameter_name,
                  size_t parameter_size, void *parameter_value,
                  size_t *parameter_size_ret) {
  if (!sagr_cl_valid_platform(platform)) {
    return CL_INVALID_PLATFORM;
  }
  switch (parameter_name) {
    case CL_PLATFORM_PROFILE:
      return sagr_cl_copy_string("FULL_PROFILE", parameter_size,
                                 parameter_value, parameter_size_ret);
    case CL_PLATFORM_VERSION:
      return sagr_cl_copy_string("OpenCL 1.2 self-amdgpu-sim CP27",
                                 parameter_size, parameter_value,
                                 parameter_size_ret);
    case CL_PLATFORM_NAME:
      return sagr_cl_copy_string("self-amdgpu gem5 OpenCL", parameter_size,
                                 parameter_value, parameter_size_ret);
    case CL_PLATFORM_VENDOR:
      return sagr_cl_copy_string("self-amdgpu-runtime", parameter_size,
                                 parameter_value, parameter_size_ret);
    case CL_PLATFORM_EXTENSIONS:
      return sagr_cl_copy_string("", parameter_size, parameter_value,
                                 parameter_size_ret);
    default:
      return CL_INVALID_VALUE;
  }
}

SAGR_CL_EXPORT cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
               cl_uint num_entries, cl_device_id *devices,
               cl_uint *num_devices) {
  const cl_device_type known =
      CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU |
      CL_DEVICE_TYPE_ACCELERATOR | CL_DEVICE_TYPE_CUSTOM;
  if (!sagr_cl_valid_platform(platform)) {
    return CL_INVALID_PLATFORM;
  }
  if (device_type == 0U ||
      (device_type != CL_DEVICE_TYPE_ALL && (device_type & ~known) != 0U)) {
    return CL_INVALID_DEVICE_TYPE;
  }
  if ((devices == NULL && num_devices == NULL) ||
      (devices != NULL && num_entries == 0U)) {
    return CL_INVALID_VALUE;
  }
  if (device_type != CL_DEVICE_TYPE_ALL &&
      (device_type & (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_DEFAULT)) == 0U) {
    if (num_devices != NULL) {
      *num_devices = 0U;
    }
    return CL_DEVICE_NOT_FOUND;
  }
  if (devices != NULL) {
    devices[0] = &sagr_cl_device;
  }
  if (num_devices != NULL) {
    *num_devices = 1U;
  }
  return CL_SUCCESS;
}

#define DEVICE_SCALAR(name, type, value)                                      \
  case name: {                                                                 \
    const type result = (value);                                               \
    return sagr_cl_copy_info(&result, sizeof(result), parameter_size,          \
                             parameter_value, parameter_size_ret);             \
  }

SAGR_CL_EXPORT cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id device, cl_device_info parameter_name,
                size_t parameter_size, void *parameter_value,
                size_t *parameter_size_ret) {
  if (!sagr_cl_valid_device(device)) {
    return CL_INVALID_DEVICE;
  }
  switch (parameter_name) {
    DEVICE_SCALAR(CL_DEVICE_TYPE, cl_device_type, CL_DEVICE_TYPE_GPU)
    DEVICE_SCALAR(CL_DEVICE_VENDOR_ID, cl_uint, 0x1002U)
    DEVICE_SCALAR(CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint, 1U)
    DEVICE_SCALAR(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint, 3U)
    case CL_DEVICE_MAX_WORK_ITEM_SIZES: {
      const size_t sizes[3] = {SAGR_CL_MAX_WORK_ITEM_SIZE,
                               SAGR_CL_MAX_WORK_ITEM_SIZE,
                               SAGR_CL_MAX_WORK_ITEM_SIZE};
      return sagr_cl_copy_info(sizes, sizeof(sizes), parameter_size,
                               parameter_value, parameter_size_ret);
    }
    DEVICE_SCALAR(CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t,
                  SAGR_CL_MAX_WORK_GROUP_SIZE)
    DEVICE_SCALAR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, cl_uint, 1U)
    DEVICE_SCALAR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, cl_uint, 1U)
    DEVICE_SCALAR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint, 1U)
    DEVICE_SCALAR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint, 1U)
    DEVICE_SCALAR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint, 1U)
    DEVICE_SCALAR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint, 0U)
    DEVICE_SCALAR(CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint, 1000U)
    DEVICE_SCALAR(CL_DEVICE_ADDRESS_BITS, cl_uint, 64U)
    DEVICE_SCALAR(CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong,
                  (cl_ulong)SAGR_MEMORY_MAX_SINGLE_ALLOCATION_BYTES)
    DEVICE_SCALAR(CL_DEVICE_IMAGE_SUPPORT, cl_bool, CL_FALSE)
    DEVICE_SCALAR(CL_DEVICE_MAX_PARAMETER_SIZE, size_t,
                  (size_t)SAGR_GENERIC_MAX_KERNARG_BYTES)
    DEVICE_SCALAR(CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint, 32768U)
    DEVICE_SCALAR(CL_DEVICE_SINGLE_FP_CONFIG, cl_device_fp_config,
                  CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN)
    DEVICE_SCALAR(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, cl_device_mem_cache_type,
                  CL_NONE)
    DEVICE_SCALAR(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint, 64U)
    DEVICE_SCALAR(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong, 0U)
    DEVICE_SCALAR(CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong,
                  (cl_ulong)SAGR_MEMORY_MAX_TOTAL_LIVE_BYTES)
    DEVICE_SCALAR(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong, 0U)
    DEVICE_SCALAR(CL_DEVICE_MAX_CONSTANT_ARGS, cl_uint, 0U)
    DEVICE_SCALAR(CL_DEVICE_LOCAL_MEM_TYPE, cl_device_local_mem_type,
                  CL_LOCAL)
    DEVICE_SCALAR(CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong, 65536U)
    DEVICE_SCALAR(CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool, CL_FALSE)
    DEVICE_SCALAR(CL_DEVICE_HOST_UNIFIED_MEMORY, cl_bool, CL_FALSE)
    DEVICE_SCALAR(CL_DEVICE_PROFILING_TIMER_RESOLUTION, size_t, 1U)
    DEVICE_SCALAR(CL_DEVICE_ENDIAN_LITTLE, cl_bool, CL_TRUE)
    DEVICE_SCALAR(CL_DEVICE_AVAILABLE, cl_bool, CL_TRUE)
    DEVICE_SCALAR(CL_DEVICE_COMPILER_AVAILABLE, cl_bool, CL_TRUE)
    DEVICE_SCALAR(CL_DEVICE_LINKER_AVAILABLE, cl_bool, CL_TRUE)
    DEVICE_SCALAR(CL_DEVICE_EXECUTION_CAPABILITIES,
                  cl_device_exec_capabilities, CL_EXEC_KERNEL)
    DEVICE_SCALAR(CL_DEVICE_QUEUE_PROPERTIES, cl_command_queue_properties, 0U)
    DEVICE_SCALAR(CL_DEVICE_PLATFORM, cl_platform_id, &sagr_cl_platform)
    case CL_DEVICE_NAME:
      return sagr_cl_copy_string("gfx950 gem5 (self-amdgpu)", parameter_size,
                                 parameter_value, parameter_size_ret);
    case CL_DEVICE_VENDOR:
      return sagr_cl_copy_string("self-amdgpu-runtime", parameter_size,
                                 parameter_value, parameter_size_ret);
    case CL_DRIVER_VERSION:
      return sagr_cl_copy_string("0.8.0-ccl-v1", parameter_size,
                                 parameter_value, parameter_size_ret);
    case CL_DEVICE_PROFILE:
      return sagr_cl_copy_string("FULL_PROFILE", parameter_size,
                                 parameter_value, parameter_size_ret);
    case CL_DEVICE_VERSION:
      return sagr_cl_copy_string("OpenCL 1.2 self-amdgpu-sim CP27",
                                 parameter_size, parameter_value,
                                 parameter_size_ret);
    case CL_DEVICE_OPENCL_C_VERSION:
      return sagr_cl_copy_string("OpenCL C 2.0 self-amdgpu clang",
                                 parameter_size, parameter_value,
                                 parameter_size_ret);
    case CL_DEVICE_EXTENSIONS:
      return sagr_cl_copy_string("", parameter_size, parameter_value,
                                 parameter_size_ret);
    default:
      return CL_INVALID_VALUE;
  }
}

#undef DEVICE_SCALAR

static cl_int validate_context_properties(const cl_context_properties *values) {
  size_t index;
  if (values == NULL) {
    return CL_SUCCESS;
  }
  for (index = 0U; index < 32U; index += 2U) {
    if (values[index] == 0) {
      return CL_SUCCESS;
    }
    if (values[index] != CL_CONTEXT_PLATFORM ||
        (cl_platform_id)(uintptr_t)values[index + 1U] != &sagr_cl_platform) {
      return CL_INVALID_PROPERTY;
    }
  }
  return CL_INVALID_PROPERTY;
}

SAGR_CL_EXPORT cl_context CL_API_CALL clCreateContext(
    const cl_context_properties *properties, cl_uint num_devices,
    const cl_device_id *devices,
    void(CL_CALLBACK *notify)(const char *, const void *, size_t, void *),
    void *user_data, cl_int *error_ret) {
  cl_context context;
  cl_int status = validate_context_properties(properties);
  if (status != CL_SUCCESS) {
    set_error(error_ret, status);
    return NULL;
  }
  if (num_devices != 1U || devices == NULL ||
      !sagr_cl_valid_device(devices[0])) {
    set_error(error_ret, num_devices == 0U || devices == NULL
                             ? CL_INVALID_VALUE
                             : CL_INVALID_DEVICE);
    return NULL;
  }
  if (notify == NULL && user_data != NULL) {
    set_error(error_ret, CL_INVALID_VALUE);
    return NULL;
  }
  context = (cl_context)calloc(1U, sizeof(*context));
  if (context == NULL) {
    set_error(error_ret, CL_OUT_OF_HOST_MEMORY);
    return NULL;
  }
  if (pthread_mutex_init(&context->mutex, NULL) != 0) {
    free(context);
    set_error(error_ret, CL_OUT_OF_HOST_MEMORY);
    return NULL;
  }
  context->magic = SAGR_CL_CONTEXT_MAGIC;
  atomic_init(&context->references, 1U);
  context->device = devices[0];
  context->notify = notify;
  context->notify_data = user_data;
  sagr_cl_simulator_init(&context->simulator);
  set_error(error_ret, CL_SUCCESS);
  return context;
}

SAGR_CL_EXPORT cl_command_queue CL_API_CALL clCreateCommandQueue(
    cl_context context, cl_device_id device,
    cl_command_queue_properties properties, cl_int *error_ret) {
  cl_command_queue queue;
  if (!sagr_cl_valid_context(context)) {
    set_error(error_ret, CL_INVALID_CONTEXT);
    return NULL;
  }
  if (!sagr_cl_valid_device(device) || device != context->device) {
    set_error(error_ret, CL_INVALID_DEVICE);
    return NULL;
  }
  if (properties != 0U) {
    set_error(error_ret, CL_INVALID_QUEUE_PROPERTIES);
    return NULL;
  }
  queue = (cl_command_queue)calloc(1U, sizeof(*queue));
  if (queue == NULL) {
    set_error(error_ret, CL_OUT_OF_HOST_MEMORY);
    return NULL;
  }
  queue->magic = SAGR_CL_QUEUE_MAGIC;
  atomic_init(&queue->references, 1U);
  queue->context = context;
  queue->device = device;
  sagr_cl_context_retain_internal(context);
  set_error(error_ret, CL_SUCCESS);
  return queue;
}

SAGR_CL_EXPORT cl_program CL_API_CALL clCreateProgramWithSource(
    cl_context context, cl_uint count, const char **strings,
    const size_t *lengths, cl_int *error_ret) {
  cl_program program;
  size_t total = 0U;
  cl_uint index;
  if (!sagr_cl_valid_context(context)) {
    set_error(error_ret, CL_INVALID_CONTEXT);
    return NULL;
  }
  if (count == 0U || strings == NULL) {
    set_error(error_ret, CL_INVALID_VALUE);
    return NULL;
  }
  for (index = 0U; index < count; ++index) {
    const size_t length =
        strings[index] == NULL
            ? 0U
            : (lengths != NULL && lengths[index] != 0U
                   ? lengths[index]
                   : strlen(strings[index]));
    if (strings[index] == NULL || length > (size_t)SAGR_CL_MAX_SOURCE_BYTES ||
        total > (size_t)SAGR_CL_MAX_SOURCE_BYTES - length) {
      set_error(error_ret, CL_INVALID_VALUE);
      return NULL;
    }
    total += length;
  }
  if (total == 0U) {
    set_error(error_ret, CL_INVALID_VALUE);
    return NULL;
  }
  program = (cl_program)calloc(1U, sizeof(*program));
  if (program == NULL) {
    set_error(error_ret, CL_OUT_OF_HOST_MEMORY);
    return NULL;
  }
  program->source = (char *)malloc(total + 1U);
  if (program->source == NULL) {
    free(program);
    set_error(error_ret, CL_OUT_OF_HOST_MEMORY);
    return NULL;
  }
  total = 0U;
  for (index = 0U; index < count; ++index) {
    const size_t length =
        lengths != NULL && lengths[index] != 0U ? lengths[index]
                                                : strlen(strings[index]);
    memcpy(program->source + total, strings[index], length);
    total += length;
  }
  program->source[total] = '\0';
  program->source_size = total;
  program->magic = SAGR_CL_PROGRAM_MAGIC;
  atomic_init(&program->references, 1U);
  program->context = context;
  program->build_status = CL_BUILD_NONE;
  sagr_cl_context_retain_internal(context);
  set_error(error_ret, CL_SUCCESS);
  return program;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clBuildProgram(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, void(CL_CALLBACK *notify)(cl_program, void *),
    void *user_data) {
  cl_int result;
  if (!sagr_cl_valid_program(program)) {
    return CL_INVALID_PROGRAM;
  }
  if ((num_devices == 0U && device_list != NULL) ||
      (num_devices != 0U && device_list == NULL)) {
    return CL_INVALID_VALUE;
  }
  if (num_devices > 1U ||
      (num_devices == 1U &&
       (!sagr_cl_valid_device(device_list[0]) ||
        device_list[0] != program->context->device))) {
    return CL_INVALID_DEVICE;
  }
  if (notify == NULL && user_data != NULL) {
    return CL_INVALID_VALUE;
  }
  result = sagr_cl_compile_program(program, options);
  if (notify != NULL) {
    notify(program, user_data);
  }
  return result;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clGetProgramBuildInfo(
    cl_program program, cl_device_id device, cl_program_build_info parameter_name,
    size_t parameter_size, void *parameter_value,
    size_t *parameter_size_ret) {
  if (!sagr_cl_valid_program(program)) {
    return CL_INVALID_PROGRAM;
  }
  if (!sagr_cl_valid_device(device) || device != program->context->device) {
    return CL_INVALID_DEVICE;
  }
  switch (parameter_name) {
    case CL_PROGRAM_BUILD_STATUS:
      return sagr_cl_copy_info(&program->build_status,
                               sizeof(program->build_status), parameter_size,
                               parameter_value, parameter_size_ret);
    case CL_PROGRAM_BUILD_OPTIONS:
      return sagr_cl_copy_string(program->build_options != NULL
                                     ? program->build_options
                                     : "",
                                 parameter_size, parameter_value,
                                 parameter_size_ret);
    case CL_PROGRAM_BUILD_LOG:
      return sagr_cl_copy_string(program->build_log != NULL
                                     ? program->build_log
                                     : "",
                                 parameter_size, parameter_value,
                                 parameter_size_ret);
    case CL_PROGRAM_BINARY_TYPE: {
      const cl_program_binary_type type =
          program->build_status == CL_BUILD_SUCCESS
              ? CL_PROGRAM_BINARY_TYPE_EXECUTABLE
              : CL_PROGRAM_BINARY_TYPE_NONE;
      return sagr_cl_copy_info(&type, sizeof(type), parameter_size,
                               parameter_value, parameter_size_ret);
    }
    default:
      return CL_INVALID_VALUE;
  }
}

SAGR_CL_EXPORT cl_kernel CL_API_CALL clCreateKernel(cl_program program,
                                                    const char *kernel_name,
                                                    cl_int *error_ret) {
  cl_kernel kernel;
  sagr_status_t status;
  if (!sagr_cl_valid_program(program)) {
    set_error(error_ret, CL_INVALID_PROGRAM);
    return NULL;
  }
  if (program->build_status != CL_BUILD_SUCCESS || program->image == NULL) {
    set_error(error_ret, CL_INVALID_PROGRAM_EXECUTABLE);
    return NULL;
  }
  if (kernel_name == NULL || kernel_name[0] == '\0') {
    set_error(error_ret, CL_INVALID_VALUE);
    return NULL;
  }
  kernel = (cl_kernel)calloc(1U, sizeof(*kernel));
  if (kernel == NULL) {
    set_error(error_ret, CL_OUT_OF_HOST_MEMORY);
    return NULL;
  }
  memset(&kernel->info, 0, sizeof(kernel->info));
  status = sagr_code_object_get_kernel(&program->code_object, kernel_name,
                                       &kernel->info,
                                       (uint32_t)sizeof(kernel->info));
  if (status != SAGR_STATUS_SUCCESS) {
    free(kernel);
    set_error(error_ret, CL_INVALID_KERNEL_NAME);
    return NULL;
  }
  if (kernel->info.visible_arg_count > SAGR_CODE_OBJECT_MAX_ARGS) {
    free(kernel);
    set_error(error_ret, CL_INVALID_KERNEL_DEFINITION);
    return NULL;
  }
  kernel->magic = SAGR_CL_KERNEL_MAGIC;
  atomic_init(&kernel->references, 1U);
  kernel->program = program;
  sagr_cl_program_retain_internal(program);
  set_error(error_ret, CL_SUCCESS);
  return kernel;
}

static const sagr_code_object_arg_info_t *visible_arg(
    const sagr_code_object_kernel_info_t *kernel, cl_uint requested) {
  uint32_t index;
  cl_uint visible = 0U;
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

SAGR_CL_EXPORT cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel,
                                                 cl_uint arg_index,
                                                 size_t arg_size,
                                                 const void *arg_value) {
  const sagr_code_object_arg_info_t *metadata;
  struct sagr_cl_kernel_arg *slot;
  if (!sagr_cl_valid_kernel(kernel)) {
    return CL_INVALID_KERNEL;
  }
  metadata = visible_arg(&kernel->info, arg_index);
  if (metadata == NULL) {
    return CL_INVALID_ARG_INDEX;
  }
  if (arg_index >= SAGR_CODE_OBJECT_MAX_ARGS) {
    return CL_INVALID_ARG_INDEX;
  }
  slot = &kernel->args[arg_index];
  if (strcmp(metadata->value_kind, "global_buffer") == 0) {
    cl_mem replacement;
    if (arg_size != sizeof(cl_mem)) {
      return CL_INVALID_ARG_SIZE;
    }
    if (arg_value == NULL) {
      return CL_INVALID_ARG_VALUE;
    }
    memcpy(&replacement, arg_value, sizeof(replacement));
    if (!sagr_cl_valid_memory(replacement) ||
        replacement->context != kernel->program->context) {
      return CL_INVALID_MEM_OBJECT;
    }
    sagr_cl_memory_retain_internal(replacement);
    if (slot->is_memory && slot->memory != NULL) {
      sagr_cl_memory_release_internal(slot->memory);
    }
    memset(slot, 0, sizeof(*slot));
    slot->is_set = 1;
    slot->is_memory = 1;
    slot->size = sizeof(cl_mem);
    slot->memory = replacement;
    return CL_SUCCESS;
  }
  if (strcmp(metadata->value_kind, "by_value") != 0) {
    return CL_INVALID_OPERATION;
  }
  if (metadata->size_bytes > sizeof(slot->bytes) ||
      arg_size != metadata->size_bytes) {
    return CL_INVALID_ARG_SIZE;
  }
  if (arg_value == NULL) {
    return CL_INVALID_ARG_VALUE;
  }
  if (slot->is_memory && slot->memory != NULL) {
    sagr_cl_memory_release_internal(slot->memory);
  }
  memset(slot, 0, sizeof(*slot));
  slot->is_set = 1;
  slot->size = arg_size;
  memcpy(slot->bytes, arg_value, arg_size);
  return CL_SUCCESS;
}

SAGR_CL_EXPORT cl_mem CL_API_CALL clCreateBuffer(cl_context context,
                                                 cl_mem_flags flags,
                                                 size_t size, void *host_ptr,
                                                 cl_int *error_ret) {
  const cl_mem_flags access =
      flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY);
  const cl_mem_flags allowed = CL_MEM_READ_WRITE | CL_MEM_READ_ONLY |
                               CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR;
  cl_mem memory;
  if (!sagr_cl_valid_context(context)) {
    set_error(error_ret, CL_INVALID_CONTEXT);
    return NULL;
  }
  if ((flags & ~allowed) != 0U ||
      (access != 0U && access != CL_MEM_READ_WRITE &&
       access != CL_MEM_READ_ONLY && access != CL_MEM_WRITE_ONLY)) {
    set_error(error_ret, CL_INVALID_VALUE);
    return NULL;
  }
  if (size == 0U || size > (size_t)SAGR_MEMORY_MAX_SINGLE_ALLOCATION_BYTES) {
    set_error(error_ret, CL_INVALID_BUFFER_SIZE);
    return NULL;
  }
  if (((flags & CL_MEM_COPY_HOST_PTR) != 0U) != (host_ptr != NULL)) {
    set_error(error_ret, CL_INVALID_HOST_PTR);
    return NULL;
  }
  memory = (cl_mem)calloc(1U, sizeof(*memory));
  if (memory == NULL) {
    set_error(error_ret, CL_OUT_OF_HOST_MEMORY);
    return NULL;
  }
  if (host_ptr != NULL) {
    memory->initial_data = (uint8_t *)malloc(size);
    if (memory->initial_data == NULL) {
      free(memory);
      set_error(error_ret, CL_OUT_OF_HOST_MEMORY);
      return NULL;
    }
    memcpy(memory->initial_data, host_ptr, size);
  }
  memory->magic = SAGR_CL_MEMORY_MAGIC;
  atomic_init(&memory->references, 1U);
  memory->context = context;
  memory->flags = access == 0U ? flags | CL_MEM_READ_WRITE : flags;
  memory->size = size;
  sagr_cl_context_retain_internal(context);
  set_error(error_ret, CL_SUCCESS);
  return memory;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clEnqueueWriteBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
    size_t offset, size_t size, const void *pointer, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  return sagr_cl_enqueue_write(command_queue, buffer, blocking_write, offset,
                               size, pointer, wait_count, wait_list, event);
}

SAGR_CL_EXPORT cl_int CL_API_CALL clEnqueueReadBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
    size_t offset, size_t size, void *pointer, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  return sagr_cl_enqueue_read(command_queue, buffer, blocking_read, offset,
                              size, pointer, wait_count, wait_list, event);
}

SAGR_CL_EXPORT cl_int CL_API_CALL clEnqueueNDRangeKernel(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dimensions,
    const size_t *global_offset, const size_t *global_size,
    const size_t *local_size, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  return sagr_cl_enqueue_kernel(command_queue, kernel, work_dimensions,
                                global_offset, global_size, local_size,
                                wait_count, wait_list, event);
}

SAGR_CL_EXPORT cl_int CL_API_CALL clFlush(cl_command_queue command_queue) {
  return sagr_cl_valid_queue(command_queue) ? CL_SUCCESS
                                            : CL_INVALID_COMMAND_QUEUE;
}

SAGR_CL_EXPORT cl_int CL_API_CALL clFinish(cl_command_queue command_queue) {
  return sagr_cl_valid_queue(command_queue) ? CL_SUCCESS
                                            : CL_INVALID_COMMAND_QUEUE;
}
