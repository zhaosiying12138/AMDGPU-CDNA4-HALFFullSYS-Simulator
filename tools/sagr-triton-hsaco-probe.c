/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/code_object.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_image(const char *path, uint8_t **image, size_t *size) {
  FILE *file;
  long length;
  size_t read_size;
  if (path == NULL || image == NULL || size == NULL) return 0;
  file = fopen(path, "rb");
  if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
    if (file != NULL) fclose(file);
    return 0;
  }
  length = ftell(file);
  if (length <= 0L || fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return 0;
  }
  *size = (size_t)length;
  *image = (uint8_t *)malloc(*size);
  if (*image == NULL) {
    fclose(file);
    return 0;
  }
  read_size = fread(*image, 1U, *size, file);
  fclose(file);
  if (read_size != *size) {
    free(*image);
    *image = NULL;
    *size = 0U;
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  uint8_t *image = NULL;
  size_t image_size = 0U;
  sagr_code_object_info_t info;
  sagr_code_object_kernel_info_t kernel;
  sagr_code_object_dispatch_binding_t binding;
  const char *kernel_name = argc > 2 ? argv[2] : "vecadd";
  if (argc < 2 || argc > 3 || !read_image(argv[1], &image, &image_size)) {
    fprintf(stderr, "usage: %s IMAGE.hsaco [KERNEL]\n", argv[0]);
    return 2;
  }
  memset(&info, 0, sizeof(info));
  memset(&kernel, 0, sizeof(kernel));
  memset(&binding, 0, sizeof(binding));
  {
    const sagr_status_t validate_status = sagr_code_object_validate(
        image, image_size, &info, (uint32_t)sizeof(info));
    const sagr_status_t kernel_status =
        validate_status == SAGR_STATUS_SUCCESS
            ? sagr_code_object_get_kernel(&info, kernel_name, &kernel,
                                          (uint32_t)sizeof(kernel))
            : validate_status;
    const sagr_status_t binding_status =
        kernel_status == SAGR_STATUS_SUCCESS
            ? sagr_code_object_describe_dispatch(
                  &info, kernel_name, image_size, &binding,
                  (uint32_t)sizeof(binding))
            : kernel_status;
    if (binding_status != SAGR_STATUS_SUCCESS) {
      fprintf(stderr, "HSACO probe failed: validate=%d kernel=%d binding=%d\n",
              validate_status, kernel_status, binding_status);
      free(image);
      return 1;
    }
  }
  printf("{\"schema\":\"self-amdgpu-runtime.triton-hsaco-probe.v1\","
         "\"image_bytes\":%zu,\"target\":\"%s\","
         "\"kernel\":\"%s\",\"gfx_target\":%u,"
         "\"kernarg_bytes\":%u,\"wavefront_size\":%u,"
         "\"code_bytes\":%llu,\"metadata_only\":%s,"
         "\"execution_supported\":%s}\n",
         image_size, info.target, kernel.name, info.gfx_target,
         kernel.kernarg_segment_size, kernel.wavefront_size,
         (unsigned long long)binding.code_size,
         (binding.flags & SAGR_CODE_OBJECT_DISPATCH_FLAG_METADATA_ONLY) != 0U
             ? "true"
             : "false",
         binding.isa_supported_by_gemsim != 0U ? "true" : "false");
  free(image);
  return 0;
}
