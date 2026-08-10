/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_OPENCL_INTERNAL_H
#define SELF_AMDGPU_RUNTIME_OPENCL_INTERNAL_H

#include <CL/cl.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <self_amdgpu_runtime/code_object.h>
#include <self_amdgpu_runtime/runtime.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#if defined(__GNUC__)
#define SAGR_CL_EXPORT __attribute__((visibility("default")))
#else
#define SAGR_CL_EXPORT
#endif

#define SAGR_CL_PLATFORM_MAGIC UINT32_C(0x5343504c)
#define SAGR_CL_DEVICE_MAGIC UINT32_C(0x53434456)
#define SAGR_CL_CONTEXT_MAGIC UINT32_C(0x53434358)
#define SAGR_CL_QUEUE_MAGIC UINT32_C(0x53435155)
#define SAGR_CL_PROGRAM_MAGIC UINT32_C(0x53435052)
#define SAGR_CL_KERNEL_MAGIC UINT32_C(0x53434b52)
#define SAGR_CL_MEMORY_MAGIC UINT32_C(0x53434d45)

#define SAGR_CL_MAX_SOURCE_BYTES (UINT64_C(16) << 20)
#define SAGR_CL_MAX_BUILD_LOG_BYTES (UINT64_C(1) << 20)
#define SAGR_CL_KERNARG_OFFSET UINT64_C(64)
#define SAGR_CL_KERNARG_ALLOCATION_BYTES UINT64_C(512)
#define SAGR_CL_PROCESS_TIMEOUT_MS UINT64_C(120000)
#define SAGR_CL_STARTUP_TIMEOUT_MS UINT64_C(15000)

/* Derive a private, nonzero topology identity without consulting a system
 * UUID/ICD service. The byte form and canonical text form are kept together
 * so the spawned child and the expected handshake constraint cannot diverge. */
static inline uint64_t sagr_cl_mix_identity(uint64_t value) {
  value ^= value >> 30U;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27U;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

static inline void sagr_cl_make_job_uuid(uint64_t epoch, uint64_t process_id,
                                         uint8_t bytes[16], char text[33]) {
  static const char digits[] = "0123456789abcdef";
  uint64_t high = sagr_cl_mix_identity(epoch ^ UINT64_C(0x6a09e667f3bcc909));
  uint64_t low = sagr_cl_mix_identity(
      process_id ^ (epoch << 1U) ^ UINT64_C(0xbb67ae8584caa73b));
  size_t index;
  if (high == 0U && low == 0U) {
    low = UINT64_C(1);
  }
  for (index = 0U; index < 8U; ++index) {
    bytes[index] = (uint8_t)(high >> (index * 8U));
    bytes[index + 8U] = (uint8_t)(low >> (index * 8U));
  }
  /* Keep the conventional UUID version/variant bits while retaining the
   * complete 128-bit nonzero identity contract used by the bridge. */
  bytes[6] = (uint8_t)((bytes[6] & 0x0fU) | 0x40U);
  bytes[8] = (uint8_t)((bytes[8] & 0x3fU) | 0x80U);
  for (index = 0U; index < 16U; ++index) {
    text[index * 2U] = digits[bytes[index] >> 4U];
    text[index * 2U + 1U] = digits[bytes[index] & 0x0fU];
  }
  text[32] = '\0';
}

struct sagr_cl_simulator {
  int paths_ready;
  sagr_managed_session_t managed_session;
  sagr_instance_t instance;
  char prefix[PATH_MAX];
  char clang_path[PATH_MAX];
};

struct _cl_platform_id {
  uint32_t magic;
};

struct _cl_device_id {
  uint32_t magic;
  cl_platform_id platform;
};

struct _cl_context {
  uint32_t magic;
  atomic_uint references;
  pthread_mutex_t mutex;
  cl_device_id device;
  void(CL_CALLBACK *notify)(const char *, const void *, size_t, void *);
  void *notify_data;
  struct sagr_cl_simulator simulator;
};

struct _cl_command_queue {
  uint32_t magic;
  atomic_uint references;
  cl_context context;
  cl_device_id device;
  sagr_queue_t queue;
  sagr_queue_info_t queue_info;
};

struct _cl_program {
  uint32_t magic;
  atomic_uint references;
  cl_context context;
  char *source;
  size_t source_size;
  uint8_t *image;
  size_t image_size;
  uint8_t image_sha256[32];
  int image_sha256_valid;
  sagr_code_object_info_t code_object;
  cl_build_status build_status;
  char *build_options;
  char *build_log;
};

struct sagr_cl_kernel_arg {
  int is_set;
  int is_memory;
  size_t size;
  uint8_t bytes[8];
  cl_mem memory;
};

struct _cl_kernel {
  uint32_t magic;
  atomic_uint references;
  cl_program program;
  sagr_code_object_kernel_info_t info;
  struct sagr_cl_kernel_arg args[SAGR_CODE_OBJECT_MAX_ARGS];
};

struct _cl_mem {
  uint32_t magic;
  atomic_uint references;
  cl_context context;
  cl_mem_flags flags;
  size_t size;
  sagr_memory_t memory;
  sagr_memory_info_t memory_info;
  uint8_t *initial_data;
};

extern struct _cl_platform_id sagr_cl_platform;
extern struct _cl_device_id sagr_cl_device;

int sagr_cl_valid_platform(cl_platform_id platform);
int sagr_cl_valid_device(cl_device_id device);
int sagr_cl_valid_context(cl_context context);
int sagr_cl_valid_queue(cl_command_queue queue);
int sagr_cl_valid_program(cl_program program);
int sagr_cl_valid_kernel(cl_kernel kernel);
int sagr_cl_valid_memory(cl_mem memory);

cl_int sagr_cl_copy_info(const void *value, size_t value_size,
                         size_t parameter_size, void *parameter_value,
                         size_t *parameter_size_ret);
cl_int sagr_cl_copy_string(const char *value, size_t parameter_size,
                           void *parameter_value,
                           size_t *parameter_size_ret);
cl_int sagr_cl_status_to_error(sagr_status_t status);

void sagr_cl_context_retain_internal(cl_context context);
void sagr_cl_program_retain_internal(cl_program program);
void sagr_cl_memory_retain_internal(cl_mem memory);
void sagr_cl_context_release_internal(cl_context context);
void sagr_cl_program_release_internal(cl_program program);
void sagr_cl_memory_release_internal(cl_mem memory);

cl_int sagr_cl_compile_program(cl_program program, const char *options);
cl_int sagr_cl_ensure_queue(cl_command_queue queue);
cl_int sagr_cl_ensure_memory(cl_mem memory);

cl_int sagr_cl_enqueue_write(cl_command_queue queue, cl_mem buffer,
                             cl_bool blocking_write, size_t offset, size_t size,
                             const void *pointer, cl_uint wait_count,
                             const cl_event *wait_list, cl_event *event);
cl_int sagr_cl_enqueue_read(cl_command_queue queue, cl_mem buffer,
                            cl_bool blocking_read, size_t offset, size_t size,
                            void *pointer, cl_uint wait_count,
                            const cl_event *wait_list, cl_event *event);
cl_int sagr_cl_enqueue_kernel(cl_command_queue queue, cl_kernel kernel,
                              cl_uint work_dimensions,
                              const size_t *global_offset,
                              const size_t *global_size,
                              const size_t *local_size, cl_uint wait_count,
                              const cl_event *wait_list, cl_event *event);

void sagr_cl_simulator_init(struct sagr_cl_simulator *simulator);
cl_int sagr_cl_simulator_ensure(struct sagr_cl_simulator *simulator);
void sagr_cl_simulator_shutdown(struct sagr_cl_simulator *simulator);
int sagr_cl_spawn_and_wait(const char *path, char *const argv[],
                           char *const environment[], const char *working_dir,
                           const char *log_path, uint64_t timeout_ms,
                           int *exit_code);

#endif
