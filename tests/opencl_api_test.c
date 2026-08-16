/* SPDX-License-Identifier: GPL-3.0-or-later */

#define CL_TARGET_OPENCL_VERSION 120
#include "opencl_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SAGR_OPENCL_VECADD_SOURCE_PATH
#define SAGR_OPENCL_VECADD_SOURCE_PATH ""
#endif

#define SAGR_VECADD_HSACO_SHA256                                            \
  "314ede16940432996c9fe190115408bf42744a8ab7d0036bf07b931e39c4cb19"
#define SAGR_VECADD_HSACO_SIZE 5160U
#define SAGR_VECADD_KERNARG_SIZE 88U

static int expect_status(cl_int actual, cl_int expected, const char *label) {
  if (actual == expected) {
    return 0;
  }
  fprintf(stderr, "OpenCL API test: %s: expected %d, got %d\n", label,
          (int)expected, (int)actual);
  return 1;
}

static int expect_true(int condition, const char *label) {
  if (condition != 0) {
    return 0;
  }
  fprintf(stderr, "OpenCL API test: %s\n", label);
  return 1;
}

static char *read_source(size_t *source_size) {
  FILE *file;
  char *source;
  long file_size;
  size_t actual_size;
  if (source_size == NULL || SAGR_OPENCL_VECADD_SOURCE_PATH[0] == '\0') {
    return NULL;
  }
  file = fopen(SAGR_OPENCL_VECADD_SOURCE_PATH, "rb");
  if (file == NULL || fseek(file, 0L, SEEK_END) != 0) {
    if (file != NULL) {
      (void)fclose(file);
    }
    return NULL;
  }
  file_size = ftell(file);
  if (file_size <= 0L || fseek(file, 0L, SEEK_SET) != 0) {
    (void)fclose(file);
    return NULL;
  }
  source = (char *)malloc((size_t)file_size + 1U);
  if (source == NULL) {
    (void)fclose(file);
    return NULL;
  }
  actual_size = fread(source, 1U, (size_t)file_size, file);
  (void)fclose(file);
  if (actual_size != (size_t)file_size) {
    free(source);
    return NULL;
  }
  source[actual_size] = '\0';
  *source_size = actual_size;
  return source;
}

static void put_u64_le(uint8_t *destination, uint64_t value) {
  size_t index;
  for (index = 0U; index < sizeof(value); ++index) {
    destination[index] = (uint8_t)(value >> (index * 8U));
  }
}

static void put_u32_le(uint8_t *destination, uint32_t value) {
  size_t index;
  for (index = 0U; index < sizeof(value); ++index) {
    destination[index] = (uint8_t)(value >> (index * 8U));
  }
}

static int test_launch_geometry(void) {
  struct sagr_cl_launch_geometry geometry;
  const size_t global_1d[1] = {1024U};
  const size_t local_1d[1] = {256U};
  const size_t global_2d[2] = {64U, 32U};
  const size_t local_2d[2] = {8U, 8U};
  const size_t global_3d[3] = {32U, 16U, 8U};
  const size_t local_3d[3] = {8U, 4U, 2U};
  const size_t local_3d_two_warps[3] = {8U, 4U, 4U};
  const size_t global_boundary[2] = {(size_t)INT_MAX, 64U};
  const size_t local_boundary[2] = {1U, 64U};
  const size_t global_rounded_overflow[2] = {(size_t)INT_MAX - 1U, 64U};
  const size_t local_rounded_overflow[2] = {3U, 64U};
  const size_t global_total_overflow[3] = {65536U, 32768U, 64U};
  const size_t local_total_overflow[3] = {1U, 1U, 64U};
  const size_t local_half_wave[2] = {8U, 4U};
  const size_t global_large_local[3] = {8U, 8U, 8U};
  const size_t local_large[3] = {8U, 8U, 8U};
  const size_t offset_2d[2] = {3U, 5U};
  const size_t overflowing_offset_2d[2] = {SIZE_MAX, 0U};
  int failures = 0;

  failures += expect_status(
      sagr_cl_prepare_launch_geometry(1U, NULL, global_1d, local_1d, 256U,
                                      64U, &geometry),
      CL_SUCCESS, "prepare 1D launch geometry");
  failures += expect_true(
      geometry.global[0] == 1024U && geometry.global[1] == 1U &&
          geometry.global[2] == 1U && geometry.local[0] == 256U &&
          geometry.local[1] == 1U && geometry.local[2] == 1U &&
          geometry.num_warps == 4U && geometry.total_workgroups == 4U,
      "1D geometry must preserve inactive-axis defaults");

  failures += expect_status(
      sagr_cl_prepare_launch_geometry(2U, NULL, global_2d, local_2d, 256U,
                                      64U, &geometry),
      CL_SUCCESS, "prepare 2D launch geometry");
  failures += expect_true(
      geometry.global[0] == 64U && geometry.global[1] == 32U &&
          geometry.global[2] == 1U && geometry.local[0] == 8U &&
          geometry.local[1] == 8U && geometry.local[2] == 1U &&
          geometry.num_warps == 1U && geometry.total_workgroups == 32U &&
          geometry.dynamic_shared_memory_bytes == 0U,
      "2D geometry must use flat local size and dynamic LDS zero");

  failures += expect_status(
      sagr_cl_prepare_launch_geometry(3U, NULL, global_3d, local_3d, 256U,
                                      64U, &geometry),
      CL_SUCCESS, "prepare 3D launch geometry");
  failures += expect_true(
      geometry.num_warps == 1U && geometry.total_workgroups == 64U,
      "3D geometry must preserve all axes");
  failures += expect_status(
      sagr_cl_prepare_launch_geometry(3U, NULL, global_3d,
                                      local_3d_two_warps, 256U, 64U,
                                      &geometry),
      CL_SUCCESS, "prepare two-warp 3D launch geometry");
  failures += expect_true(geometry.num_warps == 2U &&
                              geometry.total_workgroups == 32U,
                          "3D num_warps must use the flat local size");

  failures += expect_status(
      sagr_cl_prepare_launch_geometry(2U, NULL, global_boundary,
                                      local_boundary, 256U, 64U, &geometry),
      CL_SUCCESS, "signed total-workgroup boundary");
  failures += expect_true(geometry.total_workgroups == (uint64_t)INT_MAX,
                          "signed workgroup boundary must remain exact");
  failures += expect_status(
      sagr_cl_prepare_launch_geometry(
          2U, NULL, global_rounded_overflow, local_rounded_overflow, 256U, 64U,
          &geometry),
      CL_INVALID_GLOBAL_WORK_SIZE, "reject rounded grid overflow");
  failures += expect_status(
      sagr_cl_prepare_launch_geometry(
          3U, NULL, global_total_overflow, local_total_overflow, 256U, 64U,
          &geometry),
      CL_INVALID_GLOBAL_WORK_SIZE, "reject total workgroup overflow");
  failures += expect_status(
      sagr_cl_prepare_launch_geometry(2U, NULL, global_2d, local_half_wave,
                                      256U, 64U, &geometry),
      CL_INVALID_WORK_GROUP_SIZE, "reject sub-wave flat local size");
  failures += expect_status(
      sagr_cl_prepare_launch_geometry(3U, NULL, global_large_local, local_large,
                                      1024U, 64U, &geometry),
      CL_INVALID_WORK_GROUP_SIZE, "reject flat local size above device limit");
  failures += expect_status(
      sagr_cl_prepare_launch_geometry(2U, offset_2d, global_2d, local_2d, 256U,
                                      64U, &geometry),
      CL_SUCCESS, "prepare nonzero 2D global offset");
  failures += expect_true(geometry.offset[0] == 3U &&
                              geometry.offset[1] == 5U &&
                              geometry.offset[2] == 0U,
                          "2D global offsets must preserve active axes");
  failures += expect_status(
      sagr_cl_prepare_launch_geometry(2U, overflowing_offset_2d, global_2d,
                                      local_2d, 256U, 64U, &geometry),
      CL_INVALID_GLOBAL_OFFSET, "reject overflowing 2D global offset");
  failures += expect_status(
      sagr_cl_prepare_launch_geometry(0U, NULL, global_2d, local_2d, 256U, 64U,
                                      &geometry),
      CL_INVALID_WORK_DIMENSION, "reject zero work dimensions");
  failures += expect_status(
      sagr_cl_prepare_launch_geometry(4U, NULL, global_3d, local_3d, 256U, 64U,
                                      &geometry),
      CL_INVALID_WORK_DIMENSION, "reject four work dimensions");
  return failures;
}

int main(void) {
  static const char InvalidSource[] =
      "__kernel void intentionally_broken(__global int *output) {\n"
      "  output[get_global_id(0)] = ;\n"
      "}\n";
  const char *source = InvalidSource;
  const size_t source_size = sizeof(InvalidSource) - 1U;
  cl_uint platform_count = 0U;
  cl_uint device_count = 0U;
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_program program = NULL;
  cl_program valid_program = NULL;
  cl_program invalid_program;
  cl_kernel kernel = NULL;
  cl_mem buffer_a = NULL;
  cl_mem buffer_b = NULL;
  cl_mem buffer_c = NULL;
  cl_int status = CL_SUCCESS;
  cl_build_status build_status = CL_BUILD_NONE;
  char *valid_source = NULL;
  const char *valid_source_pointer;
  char *valid_build_log = NULL;
  size_t valid_source_size = 0U;
  size_t name_size = 0U;
  size_t build_log_size = 0U;
  size_t max_work_item_sizes[3] = {0U, 0U, 0U};
  size_t max_work_group_size = 0U;
  uint8_t kernarg[SAGR_VECADD_KERNARG_SIZE];
  uint8_t expected_kernarg[SAGR_VECADD_KERNARG_SIZE];
  sagr_code_object_arg_value_t values[11];
  uint32_t written_size = 0U;
  sagr_status_t pack_status = SAGR_STATUS_INTERNAL_ERROR;
  const uint64_t address_a = UINT64_C(0x100001000);
  const uint64_t address_b = UINT64_C(0x100002000);
  const uint64_t address_c = UINT64_C(0x100003000);
  const cl_uint element_count = 1024U;
  size_t value_index;
  int valid_build_observed = 0;
  int exact_kernarg_observed = 0;
  uint8_t identity_bytes[16];
  char identity_text[33];
  size_t identity_index;
  int failures = 0;

  failures += test_launch_geometry();

  memset(identity_bytes, 0, sizeof(identity_bytes));
  memset(identity_text, 0, sizeof(identity_text));
  sagr_cl_make_job_uuid(UINT64_C(0x123456789abcdef0), UINT64_C(4242),
                        identity_bytes, identity_text);
  failures += expect_true(identity_text[32] == '\0' &&
                              strspn(identity_text,
                                     "0123456789abcdef") == 32U,
                          "job UUID text must be canonical lowercase hex");
  failures += expect_true(
      memcmp(identity_bytes, (uint8_t[16]){0}, sizeof(identity_bytes)) != 0,
      "job UUID identity must be nonzero");
  for (identity_index = 0U; identity_index < sizeof(identity_bytes);
       ++identity_index) {
    const char digits[] = "0123456789abcdef";
    failures += expect_true(
        identity_text[identity_index * 2U] ==
                digits[identity_bytes[identity_index] >> 4U] &&
            identity_text[identity_index * 2U + 1U] ==
                digits[identity_bytes[identity_index] & 0x0fU],
        "job UUID text must match expected handshake bytes");
  }

  failures += expect_status(clGetPlatformIDs(0U, NULL, NULL),
                            CL_INVALID_VALUE,
                            "platform query rejects two null outputs");
  failures += expect_status(clGetPlatformIDs(0U, &platform, NULL),
                            CL_INVALID_VALUE,
                            "platform query rejects zero-sized output");
  failures += expect_status(clGetPlatformIDs(0U, NULL, &platform_count),
                            CL_SUCCESS, "platform count query");
  failures += expect_true(platform_count == 1U,
                          "runtime must expose exactly one platform");
  if (failures != 0) {
    goto done;
  }
  failures += expect_status(clGetPlatformIDs(1U, &platform, NULL), CL_SUCCESS,
                            "platform enumeration");
  failures += expect_true(platform != NULL, "platform handle must be non-null");
  failures += expect_status(
      clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0U, NULL, &name_size),
      CL_SUCCESS, "platform name size query");
  failures += expect_true(name_size > 1U, "platform name must not be empty");

  failures += expect_status(
      clGetDeviceIDs(platform, (cl_device_type)0, 0U, NULL, &device_count),
      CL_INVALID_DEVICE_TYPE, "device query rejects an empty device type");
  failures += expect_status(
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0U, &device, NULL),
      CL_INVALID_VALUE, "device query rejects zero-sized output");
  failures += expect_status(
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0U, NULL, &device_count),
      CL_SUCCESS, "device count query");
  failures += expect_true(device_count == 1U,
                          "runtime must expose exactly one GPU device");
  if (failures != 0) {
    goto done;
  }
  failures += expect_status(
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1U, &device, NULL),
      CL_SUCCESS, "device enumeration");
  failures += expect_true(device != NULL, "device handle must be non-null");
  failures += expect_status(
      clGetDeviceInfo(device, CL_DEVICE_NAME, 0U, NULL, &name_size), CL_SUCCESS,
      "device name size query");
  failures += expect_true(name_size > 1U, "device name must not be empty");
  failures += expect_status(
      clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                      sizeof(max_work_item_sizes), max_work_item_sizes, NULL),
      CL_SUCCESS, "device work-item dimensions query");
  failures += expect_status(
      clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                      sizeof(max_work_group_size), &max_work_group_size, NULL),
      CL_SUCCESS, "device flat workgroup limit query");
  failures += expect_true(
      max_work_item_sizes[0] == SAGR_CL_MAX_WORK_ITEM_SIZE &&
          max_work_item_sizes[1] == SAGR_CL_MAX_WORK_ITEM_SIZE &&
          max_work_item_sizes[2] == SAGR_CL_MAX_WORK_ITEM_SIZE &&
          max_work_group_size == SAGR_CL_MAX_WORK_GROUP_SIZE,
      "device geometry limits must match 1D/2D/3D launch validation");

  context = clCreateContext(NULL, 0U, NULL, NULL, NULL, &status);
  failures += expect_true(context == NULL,
                          "zero-device context creation must fail");
  failures += expect_status(status, CL_INVALID_VALUE,
                            "zero-device context status");

  context = clCreateContext(NULL, 1U, &device, NULL, NULL, &status);
  failures += expect_status(status, CL_SUCCESS, "context creation");
  failures += expect_true(context != NULL, "context handle must be non-null");
  if (context == NULL || status != CL_SUCCESS) {
    goto done;
  }
  failures += expect_status(clRetainContext(context), CL_SUCCESS,
                            "context retain");
  failures += expect_status(clReleaseContext(context), CL_SUCCESS,
                            "first context release");

  invalid_program =
      clCreateProgramWithSource(context, 0U, NULL, NULL, &status);
  failures += expect_true(invalid_program == NULL,
                          "empty source program creation must fail");
  failures += expect_status(status, CL_INVALID_VALUE,
                            "empty source program status");

  program = clCreateProgramWithSource(context, 1U, &source, &source_size,
                                      &status);
  failures += expect_status(status, CL_SUCCESS, "program creation");
  failures += expect_true(program != NULL, "program handle must be non-null");
  if (program == NULL || status != CL_SUCCESS) {
    goto done;
  }
  failures += expect_status(clRetainProgram(program), CL_SUCCESS,
                            "program retain");
  failures += expect_status(clReleaseProgram(program), CL_SUCCESS,
                            "first program release");

  status = clBuildProgram(program, 1U, &device, NULL, NULL, NULL);
  failures += expect_status(status, CL_BUILD_PROGRAM_FAILURE,
                            "invalid source build failure");
  failures += expect_status(
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                            sizeof(build_status), &build_status, NULL),
      CL_SUCCESS, "program build status query");
  failures += expect_true(build_status == CL_BUILD_ERROR,
                          "failed program must report CL_BUILD_ERROR");
  failures += expect_status(
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0U, NULL,
                            &build_log_size),
      CL_SUCCESS, "program build log size query");
  failures += expect_true(build_log_size > 1U,
                          "failed program must provide a build log");

  valid_source = read_source(&valid_source_size);
  failures += expect_true(valid_source != NULL,
                          "exact vecadd source must be readable");
  if (valid_source == NULL) {
    goto done;
  }
  valid_source_pointer = valid_source;
  valid_program = clCreateProgramWithSource(
      context, 1U, &valid_source_pointer, &valid_source_size, &status);
  failures += expect_status(status, CL_SUCCESS,
                            "valid vecadd program creation");
  failures += expect_true(valid_program != NULL,
                          "valid vecadd program must be non-null");
  if (valid_program == NULL || status != CL_SUCCESS) {
    goto done;
  }
  status = clBuildProgram(valid_program, 1U, &device, NULL, NULL, NULL);
  failures += expect_status(status, CL_SUCCESS, "valid vecadd source build");
  if (status != CL_SUCCESS) {
    goto done;
  }
  valid_build_observed = 1;
  failures += expect_status(
      clGetProgramBuildInfo(valid_program, device, CL_PROGRAM_BUILD_LOG, 0U,
                            NULL, &build_log_size),
      CL_SUCCESS, "valid build log size query");
  valid_build_log = (char *)malloc(build_log_size);
  failures += expect_true(valid_build_log != NULL,
                          "valid build log allocation");
  if (valid_build_log == NULL) {
    goto done;
  }
  failures += expect_status(
      clGetProgramBuildInfo(valid_program, device, CL_PROGRAM_BUILD_LOG,
                            build_log_size, valid_build_log, NULL),
      CL_SUCCESS, "valid build log query");
  failures += expect_true(strstr(valid_build_log, "clang_argv=") != NULL,
                          "build log must retain compiler argv");
  failures += expect_true(strstr(valid_build_log, "hsaco_size=5160") != NULL,
                          "build log must retain exact HSACO size");
  failures += expect_true(
      strstr(valid_build_log, "hsaco_sha256=" SAGR_VECADD_HSACO_SHA256) != NULL,
      "build log must retain exact HSACO SHA256");
  failures += expect_true(valid_program->image_size == SAGR_VECADD_HSACO_SIZE,
                          "program must retain the exact 5160-byte image");

  kernel = clCreateKernel(valid_program, "vecadd", &status);
  failures += expect_status(status, CL_SUCCESS, "vecadd kernel creation");
  failures += expect_true(kernel != NULL, "vecadd kernel must be non-null");
  if (kernel == NULL || status != CL_SUCCESS) {
    goto done;
  }
  failures += expect_true(
      kernel->info.kernarg_segment_size == SAGR_VECADD_KERNARG_SIZE &&
          kernel->info.kernarg_segment_align == 8U &&
          kernel->info.arg_count == 11U &&
          kernel->info.visible_arg_count == 4U &&
          kernel->info.hidden_arg_count == 7U &&
          kernel->info.wavefront_size == 64U,
      "vecadd metadata must match the audited 88-byte ABI");
  failures += expect_true(
      strcmp(kernel->info.args[7].value_kind, "hidden_none") == 0 &&
          strcmp(kernel->info.args[8].value_kind, "hidden_none") == 0 &&
          strcmp(kernel->info.args[9].value_kind, "hidden_none") == 0 &&
          strcmp(kernel->info.args[10].value_kind, "hidden_none") == 0,
      "vecadd trailing hidden_none metadata must be preserved");
  if (failures != 0) {
    goto done;
  }

  buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, 4096U, NULL, &status);
  failures += expect_status(status, CL_SUCCESS, "buffer A creation");
  buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, 4096U, NULL, &status);
  failures += expect_status(status, CL_SUCCESS, "buffer B creation");
  buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 4096U, NULL, &status);
  failures += expect_status(status, CL_SUCCESS, "buffer C creation");
  if (buffer_a == NULL || buffer_b == NULL || buffer_c == NULL ||
      failures != 0) {
    goto done;
  }
  failures += expect_status(
      clSetKernelArg(kernel, 0U, sizeof(buffer_a), &buffer_a), CL_SUCCESS,
      "set vecadd A argument");
  failures += expect_status(
      clSetKernelArg(kernel, 1U, sizeof(buffer_b), &buffer_b), CL_SUCCESS,
      "set vecadd B argument");
  failures += expect_status(
      clSetKernelArg(kernel, 2U, sizeof(buffer_c), &buffer_c), CL_SUCCESS,
      "set vecadd C argument");
  failures += expect_status(
      clSetKernelArg(kernel, 3U, sizeof(element_count), &element_count),
      CL_SUCCESS, "set vecadd n argument");
  if (failures != 0) {
    goto done;
  }

  memset(values, 0, sizeof(values));
  for (value_index = 0U; value_index < 11U; ++value_index) {
    values[value_index].struct_size =
        (uint32_t)sizeof(values[value_index]);
    values[value_index].arg_index = kernel->info.args[value_index].index;
  }
  values[0].value = address_a;
  values[1].value = address_b;
  values[2].value = address_c;
  values[3].value = element_count;
  memset(kernarg, 0xa5, sizeof(kernarg));
  memset(expected_kernarg, 0, sizeof(expected_kernarg));
  put_u64_le(expected_kernarg, address_a);
  put_u64_le(expected_kernarg + 8U, address_b);
  put_u64_le(expected_kernarg + 16U, address_c);
  put_u32_le(expected_kernarg + 24U, element_count);
  pack_status = sagr_code_object_pack_kernarg(
      &kernel->info, values, 11U, kernarg, (uint32_t)sizeof(kernarg),
      &written_size);
  failures += expect_true(pack_status == SAGR_STATUS_SUCCESS,
                          "pack exact vecadd kernarg");
  exact_kernarg_observed =
      pack_status == SAGR_STATUS_SUCCESS &&
      written_size == SAGR_VECADD_KERNARG_SIZE &&
      memcmp(kernarg, expected_kernarg, sizeof(kernarg)) == 0;
  failures += expect_true(
      exact_kernarg_observed,
      "kernarg must be pointers@0/8/16, n@24, and zero through byte 87");

done:
  if (kernel != NULL) {
    failures += expect_status(clReleaseKernel(kernel), CL_SUCCESS,
                              "kernel release");
  }
  if (buffer_c != NULL) {
    failures += expect_status(clReleaseMemObject(buffer_c), CL_SUCCESS,
                              "buffer C release");
  }
  if (buffer_b != NULL) {
    failures += expect_status(clReleaseMemObject(buffer_b), CL_SUCCESS,
                              "buffer B release");
  }
  if (buffer_a != NULL) {
    failures += expect_status(clReleaseMemObject(buffer_a), CL_SUCCESS,
                              "buffer A release");
  }
  if (valid_program != NULL) {
    failures += expect_status(clReleaseProgram(valid_program), CL_SUCCESS,
                              "valid program release");
  }
  if (program != NULL) {
    failures += expect_status(clReleaseProgram(program), CL_SUCCESS,
                              "final program release");
  }
  if (context != NULL) {
    failures += expect_status(clReleaseContext(context), CL_SUCCESS,
                              "final context release");
  }
  free(valid_build_log);
  free(valid_source);
  printf("{\"schema\":\"self-amdgpu-runtime.opencl-api-test.v1\","
         "\"platforms\":%u,\"devices\":%u,"
         "\"invalid_build_observed\":%s,\"valid_build_observed\":%s,"
         "\"exact_kernarg_observed\":%s,\"failures\":%d}\n",
         platform_count, device_count,
         build_status == CL_BUILD_ERROR ? "true" : "false",
         valid_build_observed ? "true" : "false",
         exact_kernarg_observed ? "true" : "false", failures);
  return failures == 0 ? 0 : 1;
}
