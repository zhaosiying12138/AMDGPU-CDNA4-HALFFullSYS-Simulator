/* SPDX-License-Identifier: GPL-3.0-or-later */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SAGR_OPENCL_VECADD_SOURCE_PATH
#define SAGR_OPENCL_VECADD_SOURCE_PATH ""
#endif

#define SAGR_VECADD_COUNT 1024U
#define SAGR_VECADD_LOCAL_SIZE 256U

static const char EmbeddedVecaddSource[] =
    "__kernel void\n"
    "vecadd(__global const float *a, __global const float *b, "
    "__global float *c, const uint n)\n"
    "{\n"
    "    const size_t index = get_global_id(0);\n"
    "    if (index < n)\n"
    "        c[index] = a[index] + b[index];\n"
    "}\n";

typedef struct vecadd_result {
  const char *stage;
  const char *source_origin;
  cl_int status;
  int source_compiled;
  int dispatch_completed;
  int output_correct;
  size_t mismatch_count;
  double max_abs_error;
} vecadd_result_t;

static char *read_source_file(const char *path, size_t *source_size) {
  FILE *file;
  long file_size;
  size_t size;
  size_t bytes_read;
  char *source;

  if (path == NULL || path[0] == '\0' || source_size == NULL) {
    return NULL;
  }
  *source_size = 0U;
  file = fopen(path, "rb");
  if (file == NULL) {
    return NULL;
  }
  if (fseek(file, 0L, SEEK_END) != 0) {
    (void)fclose(file);
    return NULL;
  }
  file_size = ftell(file);
  if (file_size < 0L || fseek(file, 0L, SEEK_SET) != 0) {
    (void)fclose(file);
    return NULL;
  }
  size = (size_t)file_size;
  if (size == SIZE_MAX) {
    (void)fclose(file);
    return NULL;
  }
  source = (char *)malloc(size + 1U);
  if (source == NULL) {
    (void)fclose(file);
    return NULL;
  }
  bytes_read = fread(source, 1U, size, file);
  (void)fclose(file);
  if (bytes_read != size) {
    free(source);
    return NULL;
  }
  source[size] = '\0';
  *source_size = size;
  return source;
}

static void print_build_log(cl_program program, cl_device_id device) {
  size_t log_size = 0U;
  char *log;

  if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0U, NULL,
                            &log_size) != CL_SUCCESS ||
      log_size == 0U) {
    return;
  }
  log = (char *)malloc(log_size + 1U);
  if (log == NULL) {
    return;
  }
  if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                            log, NULL) == CL_SUCCESS) {
    log[log_size] = '\0';
    fprintf(stderr, "OpenCL build log:\n%s\n", log);
  }
  free(log);
}

static int release_object(const char *name, cl_int status,
                          vecadd_result_t *result) {
  if (status == CL_SUCCESS) {
    return 1;
  }
  fprintf(stderr, "OpenCL vecadd: %s failed with status %d\n", name,
          (int)status);
  if (result->status == CL_SUCCESS) {
    result->stage = name;
    result->status = status;
  }
  return 0;
}

static void print_result(const vecadd_result_t *result) {
  printf("{\"schema\":\"self-amdgpu-runtime.opencl-vecadd.v1\","
         "\"kernel\":\"vecadd\",\"n\":%u,\"global_size\":%u,"
         "\"local_size\":%u,\"source_origin\":\"%s\","
         "\"source_compiled\":%s,\"gem5_execution\":%s,"
         "\"output_correct\":%s,\"max_abs_error\":%.9g,"
         "\"mismatch_count\":%zu,\"fallback_count\":0,"
         "\"status\":%d,\"stage\":\"%s\"}\n",
         SAGR_VECADD_COUNT, SAGR_VECADD_COUNT, SAGR_VECADD_LOCAL_SIZE,
         result->source_origin, result->source_compiled ? "true" : "false",
         result->dispatch_completed ? "true" : "false",
         result->output_correct ? "true" : "false", result->max_abs_error,
         result->mismatch_count, (int)result->status, result->stage);
}

int main(int argc, char **argv) {
  const char *requested_path = NULL;
  const char *source_origin = "embedded";
  const char *source;
  char *owned_source = NULL;
  size_t source_size = 0U;
  cl_uint platform_count = 0U;
  cl_uint device_count = 0U;
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem buffer_a = NULL;
  cl_mem buffer_b = NULL;
  cl_mem buffer_c = NULL;
  float *host_a = NULL;
  float *host_b = NULL;
  float *host_c = NULL;
  const size_t buffer_bytes = (size_t)SAGR_VECADD_COUNT * sizeof(float);
  const size_t global_size = (size_t)SAGR_VECADD_COUNT;
  const size_t local_size = (size_t)SAGR_VECADD_LOCAL_SIZE;
  const cl_uint element_count = (cl_uint)SAGR_VECADD_COUNT;
  vecadd_result_t result = {"arguments", "embedded", CL_SUCCESS, 0, 0, 0,
                            0U, 0.0};
  cl_int status;
  size_t index;

  if (argc > 2) {
    fprintf(stderr, "usage: %s [vecadd.cl]\n", argv[0]);
    result.status = CL_INVALID_VALUE;
    print_result(&result);
    return 1;
  }

  if (argc == 2) {
    requested_path = argv[1];
    source_origin = "argument";
  } else {
    requested_path = getenv("SAGR_OPENCL_VECADD_SOURCE_PATH");
    if (requested_path != NULL && requested_path[0] != '\0') {
      source_origin = "environment";
    } else if (SAGR_OPENCL_VECADD_SOURCE_PATH[0] != '\0') {
      requested_path = SAGR_OPENCL_VECADD_SOURCE_PATH;
      source_origin = "configured";
    } else {
      requested_path = NULL;
    }
  }

  if (requested_path != NULL) {
    owned_source = read_source_file(requested_path, &source_size);
    if (owned_source == NULL) {
      fprintf(stderr, "OpenCL vecadd: could not read source file '%s'\n",
              requested_path);
      result.stage = "read_source";
      result.status = CL_INVALID_VALUE;
      result.source_origin = source_origin;
      print_result(&result);
      return 1;
    }
    source = owned_source;
  } else {
    source = EmbeddedVecaddSource;
    source_size = sizeof(EmbeddedVecaddSource) - 1U;
  }
  result.source_origin = source_origin;

  host_a = (float *)malloc(buffer_bytes);
  host_b = (float *)malloc(buffer_bytes);
  host_c = (float *)malloc(buffer_bytes);
  if (host_a == NULL || host_b == NULL || host_c == NULL) {
    result.stage = "host_allocation";
    result.status = CL_OUT_OF_HOST_MEMORY;
    goto cleanup;
  }
  for (index = 0U; index < (size_t)SAGR_VECADD_COUNT; ++index) {
    host_a[index] = (float)index;
    host_b[index] = (float)(2U * (unsigned int)index + 1U);
    host_c[index] = -1.0F;
  }

#define REQUIRE_CL(call, label)                                                \
  do {                                                                         \
    status = (call);                                                           \
    if (status != CL_SUCCESS) {                                                \
      fprintf(stderr, "OpenCL vecadd: %s failed with status %d\n", (label),  \
              (int)status);                                                    \
      result.stage = (label);                                                  \
      result.status = status;                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

  REQUIRE_CL(clGetPlatformIDs(0U, NULL, &platform_count), "platform_count");
  if (platform_count == 0U) {
    result.stage = "platform_count";
    result.status = CL_DEVICE_NOT_FOUND;
    goto cleanup;
  }
  REQUIRE_CL(clGetPlatformIDs(1U, &platform, NULL), "platform");
  REQUIRE_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0U, NULL,
                            &device_count),
             "device_count");
  if (device_count == 0U) {
    result.stage = "device_count";
    result.status = CL_DEVICE_NOT_FOUND;
    goto cleanup;
  }
  REQUIRE_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1U, &device, NULL),
             "device");

  context = clCreateContext(NULL, 1U, &device, NULL, NULL, &status);
  if (context == NULL || status != CL_SUCCESS) {
    result.stage = "create_context";
    result.status = status;
    goto cleanup;
  }
  queue = clCreateCommandQueue(context, device, 0U, &status);
  if (queue == NULL || status != CL_SUCCESS) {
    result.stage = "create_queue";
    result.status = status;
    goto cleanup;
  }
  program = clCreateProgramWithSource(context, 1U, &source, &source_size,
                                      &status);
  if (program == NULL || status != CL_SUCCESS) {
    result.stage = "create_program";
    result.status = status;
    goto cleanup;
  }
  status = clBuildProgram(program, 1U, &device, NULL, NULL, NULL);
  if (status != CL_SUCCESS) {
    print_build_log(program, device);
    result.stage = "build_program";
    result.status = status;
    goto cleanup;
  }
  result.source_compiled = 1;

  kernel = clCreateKernel(program, "vecadd", &status);
  if (kernel == NULL || status != CL_SUCCESS) {
    result.stage = "create_kernel";
    result.status = status;
    goto cleanup;
  }
  buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_bytes, NULL,
                            &status);
  if (buffer_a == NULL || status != CL_SUCCESS) {
    result.stage = "create_buffer_a";
    result.status = status;
    goto cleanup;
  }
  buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_bytes, NULL,
                            &status);
  if (buffer_b == NULL || status != CL_SUCCESS) {
    result.stage = "create_buffer_b";
    result.status = status;
    goto cleanup;
  }
  buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_bytes, NULL,
                            &status);
  if (buffer_c == NULL || status != CL_SUCCESS) {
    result.stage = "create_buffer_c";
    result.status = status;
    goto cleanup;
  }

  REQUIRE_CL(clEnqueueWriteBuffer(queue, buffer_a, CL_TRUE, 0U, buffer_bytes,
                                  host_a, 0U, NULL, NULL),
             "write_a");
  REQUIRE_CL(clEnqueueWriteBuffer(queue, buffer_b, CL_TRUE, 0U, buffer_bytes,
                                  host_b, 0U, NULL, NULL),
             "write_b");
  REQUIRE_CL(clSetKernelArg(kernel, 0U, sizeof(buffer_a), &buffer_a),
             "set_arg_a");
  REQUIRE_CL(clSetKernelArg(kernel, 1U, sizeof(buffer_b), &buffer_b),
             "set_arg_b");
  REQUIRE_CL(clSetKernelArg(kernel, 2U, sizeof(buffer_c), &buffer_c),
             "set_arg_c");
  REQUIRE_CL(clSetKernelArg(kernel, 3U, sizeof(element_count), &element_count),
             "set_arg_n");
  REQUIRE_CL(clEnqueueNDRangeKernel(queue, kernel, 1U, NULL, &global_size,
                                    &local_size, 0U, NULL, NULL),
             "enqueue_kernel");
  REQUIRE_CL(clFinish(queue), "finish_kernel");
  result.dispatch_completed = 1;
  REQUIRE_CL(clEnqueueReadBuffer(queue, buffer_c, CL_TRUE, 0U, buffer_bytes,
                                 host_c, 0U, NULL, NULL),
             "read_c");
  REQUIRE_CL(clFinish(queue), "finish_read");

  for (index = 0U; index < (size_t)SAGR_VECADD_COUNT; ++index) {
    const float expected = host_a[index] + host_b[index];
    double difference = (double)host_c[index] - (double)expected;
    if (difference < 0.0) {
      difference = -difference;
    }
    if (difference > result.max_abs_error) {
      result.max_abs_error = difference;
    }
    if (memcmp(&host_c[index], &expected, sizeof(expected)) != 0) {
      ++result.mismatch_count;
    }
  }
  result.output_correct = result.mismatch_count == 0U;
  result.stage = result.output_correct ? "complete" : "oracle";
  result.status = result.output_correct ? CL_SUCCESS : CL_INVALID_VALUE;

cleanup:
#undef REQUIRE_CL
  if (buffer_c != NULL) {
    (void)release_object("release_buffer_c", clReleaseMemObject(buffer_c),
                         &result);
  }
  if (buffer_b != NULL) {
    (void)release_object("release_buffer_b", clReleaseMemObject(buffer_b),
                         &result);
  }
  if (buffer_a != NULL) {
    (void)release_object("release_buffer_a", clReleaseMemObject(buffer_a),
                         &result);
  }
  if (kernel != NULL) {
    (void)release_object("release_kernel", clReleaseKernel(kernel), &result);
  }
  if (program != NULL) {
    (void)release_object("release_program", clReleaseProgram(program),
                         &result);
  }
  if (queue != NULL) {
    (void)release_object("release_queue", clReleaseCommandQueue(queue),
                         &result);
  }
  if (context != NULL) {
    (void)release_object("release_context", clReleaseContext(context),
                         &result);
  }
  free(host_c);
  free(host_b);
  free(host_a);
  free(owned_source);
  print_result(&result);
  return result.status == CL_SUCCESS && result.output_correct ? 0 : 1;
}
