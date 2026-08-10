/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _POSIX_C_SOURCE 200809L

#include "opencl_internal.h"
#include "sha256_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int write_complete(int descriptor, const void *bytes, size_t size) {
  const uint8_t *cursor = (const uint8_t *)bytes;
  while (size != 0U) {
    const ssize_t written = write(descriptor, cursor, size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return 0;
    }
    if (written == 0) {
      return 0;
    }
    cursor += (size_t)written;
    size -= (size_t)written;
  }
  return 1;
}

static int read_file(const char *path, size_t maximum, uint8_t **bytes,
                     size_t *size) {
  struct stat state;
  uint8_t *result;
  size_t offset = 0U;
  int descriptor;
  if (path == NULL || bytes == NULL || size == NULL ||
      stat(path, &state) != 0 || !S_ISREG(state.st_mode) || state.st_size < 0 ||
      (uintmax_t)state.st_size > (uintmax_t)maximum) {
    return 0;
  }
  result = (uint8_t *)malloc((size_t)state.st_size + 1U);
  if (result == NULL) {
    return 0;
  }
  descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    free(result);
    return 0;
  }
  while (offset < (size_t)state.st_size) {
    const ssize_t count =
        read(descriptor, result + offset, (size_t)state.st_size - offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      (void)close(descriptor);
      free(result);
      return 0;
    }
    if (count == 0) {
      break;
    }
    offset += (size_t)count;
  }
  (void)close(descriptor);
  if (offset != (size_t)state.st_size) {
    free(result);
    return 0;
  }
  result[offset] = 0U;
  *bytes = result;
  *size = offset;
  return 1;
}

static char *duplicate_string(const char *value) {
  const size_t size = strlen(value) + 1U;
  char *result = (char *)malloc(size);
  if (result != NULL) {
    memcpy(result, value, size);
  }
  return result;
}

static void set_log(cl_program program, const char *message) {
  char *replacement = duplicate_string(message != NULL ? message : "");
  if (replacement != NULL) {
    free(program->build_log);
    program->build_log = replacement;
  }
}

static int make_path(char *destination, size_t destination_size,
                     const char *left, const char *right) {
  const int count = snprintf(destination, destination_size, "%s/%s", left,
                             right);
  return count > 0 && (size_t)count < destination_size;
}

static int prepare_environment(const struct sagr_cl_simulator *simulator,
                               const char *home, char storage[7][PATH_MAX + 32],
                               char *environment[8]) {
  int counts[7];
  size_t index;
  counts[0] = snprintf(storage[0], sizeof(storage[0]), "PATH=%s/bin:/usr/bin:/bin",
                       simulator->prefix);
  counts[1] = snprintf(storage[1], sizeof(storage[1]), "HOME=%s", home);
  counts[2] = snprintf(storage[2], sizeof(storage[2]), "TMPDIR=%s", home);
  counts[3] = snprintf(storage[3], sizeof(storage[3]), "XDG_CACHE_HOME=%s/cache",
                       home);
  counts[4] = snprintf(storage[4], sizeof(storage[4]), "ROCM_PATH=%s",
                       simulator->prefix);
  counts[5] = snprintf(storage[5], sizeof(storage[5]), "SOURCE_DATE_EPOCH=0");
  counts[6] = snprintf(storage[6], sizeof(storage[6]), "LC_ALL=C");
  for (index = 0U; index < 7U; ++index) {
    if (counts[index] < 0 || (size_t)counts[index] >= sizeof(storage[index])) {
      return 0;
    }
    environment[index] = storage[index];
  }
  environment[7] = NULL;
  return 1;
}

static int compiler_identity(cl_program program, const char *temporary,
                             char *resource, size_t resource_size,
                             char *const environment[]) {
  char output[PATH_MAX];
  uint8_t *bytes = NULL;
  size_t size = 0U;
  int exit_code = -1;
  char *arguments[3];
  char *newline;
  size_t prefix_size;
  if (!make_path(output, sizeof(output), temporary, "resource-dir.log")) {
    return 0;
  }
  arguments[0] = program->context->simulator.clang_path;
  arguments[1] = (char *)"-print-resource-dir";
  arguments[2] = NULL;
  if (!sagr_cl_spawn_and_wait(arguments[0], arguments, environment, temporary,
                              output, SAGR_CL_PROCESS_TIMEOUT_MS, &exit_code) ||
      exit_code != 0 ||
      !read_file(output, (size_t)SAGR_CL_MAX_BUILD_LOG_BYTES, &bytes, &size) ||
      size == 0U || size >= resource_size) {
    free(bytes);
    return 0;
  }
  memcpy(resource, bytes, size + 1U);
  free(bytes);
  newline = strpbrk(resource, "\r\n");
  if (newline != NULL) {
    *newline = '\0';
  }
  prefix_size = strlen(program->context->simulator.prefix);
  if (resource[0] != '/' ||
      strncmp(resource, program->context->simulator.prefix,
              prefix_size) != 0 ||
      resource[prefix_size] != '/') {
    return 0;
  }
  return 1;
}

static char *build_log_with_provenance(
    const uint8_t *compiler_log, size_t compiler_log_size,
    const uint8_t digest[SAGR_SHA256_BYTES], size_t image_size,
    const char *clang_path, const char *header, const char *rocm_option,
    const char *source_path, const char *image_path) {
  static const char digits[] = "0123456789abcdef";
  char digest_text[SAGR_SHA256_BYTES * 2U + 1U];
  char *result;
  int suffix_size;
  int written;
  size_t index;
  size_t separator_size = 0U;
  size_t total;
  for (index = 0U; index < SAGR_SHA256_BYTES; ++index) {
    digest_text[index * 2U] = digits[digest[index] >> 4U];
    digest_text[index * 2U + 1U] = digits[digest[index] & 15U];
  }
  digest_text[sizeof(digest_text) - 1U] = '\0';
  suffix_size = snprintf(
      NULL, 0,
      "self-amdgpu: clang_argv=%s -D ROCRTST_GPU=0x950 -x cl -target "
      "amdgcn-amd-amdhsa -include %s -mcpu=gfx950 -cl-std=CL2.0 "
      "-mcode-object-version=4 %s %s -o %s\n"
      "self-amdgpu: hsaco_size=%zu hsaco_sha256=%s\n",
      clang_path, header, rocm_option, source_path, image_path, image_size,
      digest_text);
  if (suffix_size < 0) {
    return NULL;
  }
  if (compiler_log_size != 0U && compiler_log[compiler_log_size - 1U] != '\n') {
    separator_size = 1U;
  }
  if (compiler_log_size >
      SIZE_MAX - separator_size - (size_t)suffix_size - 1U) {
    return NULL;
  }
  total = compiler_log_size + separator_size + (size_t)suffix_size;
  result = (char *)malloc(total + 1U);
  if (result == NULL) {
    return NULL;
  }
  if (compiler_log_size != 0U) {
    memcpy(result, compiler_log, compiler_log_size);
  }
  if (separator_size != 0U) {
    result[compiler_log_size] = '\n';
  }
  written = snprintf(
      result + compiler_log_size + separator_size, (size_t)suffix_size + 1U,
      "self-amdgpu: clang_argv=%s -D ROCRTST_GPU=0x950 -x cl -target "
      "amdgcn-amd-amdhsa -include %s -mcpu=gfx950 -cl-std=CL2.0 "
      "-mcode-object-version=4 %s %s -o %s\n"
      "self-amdgpu: hsaco_size=%zu hsaco_sha256=%s\n",
      clang_path, header, rocm_option, source_path, image_path, image_size,
      digest_text);
  if (written != suffix_size) {
    free(result);
    return NULL;
  }
  result[total] = '\0';
  return result;
}

cl_int sagr_cl_compile_program(cl_program program, const char *options) {
  struct sagr_cl_simulator *simulator = &program->context->simulator;
  char temporary[PATH_MAX] = "/tmp/self-amdgpu-opencl-build.XXXXXX";
  char cache[PATH_MAX];
  char source_path[PATH_MAX];
  char image_path[PATH_MAX];
  char log_path[PATH_MAX];
  char resource[PATH_MAX];
  char header[PATH_MAX];
  char rocm_option[PATH_MAX + 32];
  char environment_storage[7][PATH_MAX + 32];
  char *environment[8];
  char *arguments[20];
  uint8_t *image = NULL;
  size_t image_size = 0U;
  uint8_t *compiler_log = NULL;
  size_t compiler_log_size = 0U;
  uint8_t digest[SAGR_SHA256_BYTES];
  sagr_code_object_info_t code_object;
  char *combined_log = NULL;
  int descriptor = -1;
  int exit_code = -1;
  int index = 0;
  int rocm_option_size;
  cl_int result = CL_BUILD_PROGRAM_FAILURE;

  free(program->image);
  program->image = NULL;
  program->image_size = 0U;
  program->image_sha256_valid = 0;
  memset(&program->code_object, 0, sizeof(program->code_object));
  free(program->build_options);
  program->build_options = NULL;
  program->build_status = CL_BUILD_IN_PROGRESS;
  program->build_options = duplicate_string(options != NULL ? options : "");
  if (program->build_options == NULL) {
    program->build_status = CL_BUILD_ERROR;
    set_log(program, "self-amdgpu: out of host memory\n");
    return CL_OUT_OF_HOST_MEMORY;
  }
  if (options != NULL && options[0] != '\0') {
    program->build_status = CL_BUILD_ERROR;
    set_log(program,
            "self-amdgpu: this bounded OpenCL path accepts no build options\n");
    return CL_INVALID_BUILD_OPTIONS;
  }
  if (!simulator->paths_ready || simulator->clang_path[0] != '/') {
    program->build_status = CL_BUILD_ERROR;
    set_log(program, "self-amdgpu: isolated clang path is unavailable\n");
    return CL_BUILD_PROGRAM_FAILURE;
  }
  if (mkdtemp(temporary) == NULL || chmod(temporary, S_IRWXU) != 0 ||
      !make_path(cache, sizeof(cache), temporary, "cache") ||
      mkdir(cache, S_IRWXU) != 0 ||
      !make_path(source_path, sizeof(source_path), temporary, "program.cl") ||
      !make_path(image_path, sizeof(image_path), temporary, "program.hsaco") ||
      !make_path(log_path, sizeof(log_path), temporary, "clang.log") ||
      !prepare_environment(simulator, temporary, environment_storage,
                           environment)) {
    program->build_status = CL_BUILD_ERROR;
    set_log(program, "self-amdgpu: could not create private compiler state\n");
    return CL_BUILD_PROGRAM_FAILURE;
  }
  descriptor = open(source_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                    S_IRUSR | S_IWUSR);
  if (descriptor < 0 ||
      !write_complete(descriptor, program->source, program->source_size) ||
      close(descriptor) != 0) {
    if (descriptor >= 0) {
      (void)close(descriptor);
    }
    set_log(program, "self-amdgpu: could not materialize OpenCL source\n");
    goto cleanup;
  }
  descriptor = -1;
  rocm_option_size = snprintf(rocm_option, sizeof(rocm_option),
                              "--rocm-path=%s", simulator->prefix);
  if (!compiler_identity(program, temporary, resource, sizeof(resource),
                         environment) ||
      !make_path(header, sizeof(header), resource, "include/opencl-c.h") ||
      access(header, R_OK) != 0 || rocm_option_size < 0 ||
      (size_t)rocm_option_size >= sizeof(rocm_option)) {
    set_log(program,
            "self-amdgpu: clang resource directory escaped the local prefix\n");
    goto cleanup;
  }

  arguments[index++] = simulator->clang_path;
  arguments[index++] = (char *)"-D";
  arguments[index++] = (char *)"ROCRTST_GPU=0x950";
  arguments[index++] = (char *)"-x";
  arguments[index++] = (char *)"cl";
  arguments[index++] = (char *)"-target";
  arguments[index++] = (char *)"amdgcn-amd-amdhsa";
  arguments[index++] = (char *)"-include";
  arguments[index++] = header;
  arguments[index++] = (char *)"-mcpu=gfx950";
  arguments[index++] = (char *)"-cl-std=CL2.0";
  arguments[index++] = (char *)"-mcode-object-version=4";
  arguments[index++] = rocm_option;
  arguments[index++] = source_path;
  arguments[index++] = (char *)"-o";
  arguments[index++] = image_path;
  arguments[index] = NULL;

  if (!sagr_cl_spawn_and_wait(arguments[0], arguments, environment, temporary,
                              log_path, SAGR_CL_PROCESS_TIMEOUT_MS,
                              &exit_code)) {
    set_log(program, "self-amdgpu: isolated clang could not be executed\n");
    goto cleanup;
  }
  (void)read_file(log_path, (size_t)SAGR_CL_MAX_BUILD_LOG_BYTES, &compiler_log,
                  &compiler_log_size);
  if (exit_code != 0) {
    if (compiler_log != NULL && compiler_log_size != 0U) {
      set_log(program, (const char *)compiler_log);
    } else {
      set_log(program, "self-amdgpu: isolated clang rejected the source\n");
    }
    goto cleanup;
  }
  if (!read_file(image_path, (size_t)SAGR_CODE_OBJECT_TRANSPORT_MAX_IMAGE_BYTES,
                 &image, &image_size)) {
    set_log(program, "self-amdgpu: clang produced no bounded HSACO image\n");
    goto cleanup;
  }
  memset(&code_object, 0, sizeof(code_object));
  if (sagr_code_object_validate(image, image_size, &code_object,
                                (uint32_t)sizeof(code_object)) !=
          SAGR_STATUS_SUCCESS ||
      code_object.gfx_target != SAGR_CODE_OBJECT_TARGET_GFX950 ||
      code_object.relocation_count != 0U) {
    set_log(program,
            "self-amdgpu: clang output failed gfx950 code-object validation\n");
    goto cleanup;
  }
  sagr_sha256(image, image_size, digest);
  combined_log = build_log_with_provenance(
      compiler_log, compiler_log_size, digest, image_size,
      simulator->clang_path, header, rocm_option, source_path, image_path);
  if (combined_log == NULL) {
    set_log(program, "self-amdgpu: could not retain compiler provenance\n");
    result = CL_OUT_OF_HOST_MEMORY;
    goto cleanup;
  }
  free(program->build_log);
  program->build_log = combined_log;
  combined_log = NULL;
  program->image = image;
  program->image_size = image_size;
  image = NULL;
  memcpy(program->image_sha256, digest, sizeof(program->image_sha256));
  program->image_sha256_valid = 1;
  program->code_object = code_object;
  program->build_status = CL_BUILD_SUCCESS;
  result = CL_SUCCESS;

cleanup:
  if (result != CL_SUCCESS) {
    program->build_status = CL_BUILD_ERROR;
  }
  free(combined_log);
  free(compiler_log);
  free(image);
  if (descriptor >= 0) {
    (void)close(descriptor);
  }
  (void)unlink(source_path);
  (void)unlink(image_path);
  (void)unlink(log_path);
  if (make_path(resource, sizeof(resource), temporary, "resource-dir.log")) {
    (void)unlink(resource);
  }
  (void)rmdir(cache);
  (void)rmdir(temporary);
  return result;
}
