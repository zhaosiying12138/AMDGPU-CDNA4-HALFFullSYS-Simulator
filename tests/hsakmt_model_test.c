/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <hsakmt/hsakmtmodeliface.h>
#include <hsakmt/drm/amdgpu.h>
#include <hsakmt/linux/kfd_ioctl.h>

#define MODEL_LOCAL_MEMORY_BYTES UINT64_C(309237645312)
#define MODEL_QUEUE_CONTROL_BYTES UINT64_C(8192)
#define MODEL_BACKING_BYTES \
  (MODEL_LOCAL_MEMORY_BYTES + MODEL_QUEUE_CONTROL_BYTES)

static int expect(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "hsakmt model test: %s\n", message);
    return 1;
  }
  return 0;
}

static int check_fork_storage(const struct hsakmt_model_functions *functions,
                              ino_t parent_inode) {
  int descriptors[2];
  pid_t child;
  ino_t child_inode = 0;
  ssize_t count;
  int status = 0;
  if (pipe2(descriptors, O_CLOEXEC) != 0) {
    return 1;
  }
  child = fork();
  if (child < 0) {
    (void)close(descriptors[0]);
    (void)close(descriptors[1]);
    return 1;
  }
  if (child == 0) {
    struct stat state;
    int descriptor;
    (void)close(descriptors[0]);
    descriptor = functions->create_memfd();
    if (descriptor < 0 || fstat(descriptor, &state) != 0 ||
        write(descriptors[1], &state.st_ino, sizeof(state.st_ino)) !=
            (ssize_t)sizeof(state.st_ino)) {
      _exit(2);
    }
    _exit(0);
  }
  (void)close(descriptors[1]);
  count = read(descriptors[0], &child_inode, sizeof(child_inode));
  (void)close(descriptors[0]);
  if (waitpid(child, &status, 0) != child) {
    return 1;
  }
  return expect(count == (ssize_t)sizeof(child_inode) &&
                    WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
                    child_inode != 0 && child_inode != parent_inode,
                "forked process owns independent model storage");
}

static int check_drm_lifecycle(const struct hsakmt_model_functions *functions) {
  struct hsakmt_drm_open_render_args open_request;
  struct hsakmt_drm_device_initialize_args initialize_request;
  struct hsakmt_drm_device_get_fd_args get_fd_request;
  struct hsakmt_drm_get_marketing_name_args name_request;
  struct hsakmt_drm_query_gpu_info_args info_request;
  struct hsakmt_drm_device_deinitialize_args deinitialize_request;
  struct hsakmt_drm_close_args close_request;
  struct amdgpu_gpu_info info;
  struct stat render_state;
  uint32_t major = UINT32_C(0xa5a5a5a5);
  uint32_t minor = UINT32_C(0xa5a5a5a5);
  void *device = (void *)(uintptr_t)UINT64_C(0xa5a5a5a5a5a5a5a5);
  const char *name = (const char *)(uintptr_t)UINT64_C(0xa5a5a5a5a5a5a5a5);
  int render_fd = -1;
  int device_fd = -1;
  int untouched_fd = -7;
  int failures = 0;

  open_request.minor = 127;
  open_request.fd_out = &untouched_fd;
  errno = 0;
  failures += expect(functions->handle_drm_call(HSAKMT_DRM_OPEN_RENDER,
                                                 &open_request) == -1 &&
                         errno == EINVAL && untouched_fd == -7,
                     "out-of-range render minor fails atomically");

  open_request.minor = 128;
  open_request.fd_out = &render_fd;
  failures += expect(functions->handle_drm_call(HSAKMT_DRM_OPEN_RENDER,
                                                 &open_request) == 0 &&
                         render_fd >= 0 && fstat(render_fd, &render_state) == 0 &&
                         (fcntl(render_fd, F_GETFD) & FD_CLOEXEC) != 0,
                     "render open returns private close-on-exec model storage");

  initialize_request.fd = render_fd;
  initialize_request.major_out = &major;
  initialize_request.minor_out = &minor;
  initialize_request.dev_out = &device;
  failures += expect(functions->handle_drm_call(
                         HSAKMT_DRM_DEVICE_INITIALIZE,
                         &initialize_request) == 0 &&
                         major == 3U && minor == 57U && device != NULL,
                     "render device initializes through the generic model ABI");

  get_fd_request.dev = device;
  get_fd_request.fd_out = &device_fd;
  failures += expect(functions->handle_drm_call(HSAKMT_DRM_DEVICE_GET_FD,
                                                 &get_fd_request) == 0 &&
                         device_fd >= 0 && device_fd != render_fd &&
                         (fcntl(device_fd, F_GETFD) & FD_CLOEXEC) != 0,
                     "device owns an independent render descriptor");

  name_request.dev = device;
  name_request.name_out = &name;
  failures += expect(functions->handle_drm_call(HSAKMT_DRM_GET_MARKETING_NAME,
                                                 &name_request) == 0 &&
                         name != NULL &&
                         strcmp(name, "AMD Instinct MI350X") == 0,
                     "device exposes the pinned gfx950 marketing identity");

  memset(&info, 0xa5, sizeof(info));
  info_request.dev = device;
  info_request.info_out = &info;
  failures += expect(functions->handle_drm_call(HSAKMT_DRM_QUERY_GPU_INFO,
                                                 &info_request) == 0 &&
                         info.asic_id == 30112U && info.family_id == 160U &&
                         info.num_shader_engines == 4U &&
                         info.num_shader_arrays_per_engine == 2U &&
                         info.cu_active_number == 256U &&
                         info.vram_bit_width == 8192U,
                     "device query returns a complete gfx950 identity record");

  deinitialize_request.dev = device;
  failures += expect(functions->handle_drm_call(
                         HSAKMT_DRM_DEVICE_DEINITIALIZE,
                         &deinitialize_request) == 0,
                     "device deinitializes exactly once");
  untouched_fd = -9;
  get_fd_request.fd_out = &untouched_fd;
  errno = 0;
  failures += expect(functions->handle_drm_call(HSAKMT_DRM_DEVICE_GET_FD,
                                                 &get_fd_request) == -1 &&
                         errno == EBADF && untouched_fd == -9,
                     "retired device handle is stale and output remains unchanged");
  errno = 0;
  failures += expect(functions->handle_drm_call(
                         HSAKMT_DRM_DEVICE_DEINITIALIZE,
                         &deinitialize_request) == -1 &&
                         errno == EBADF,
                     "double device deinitialize is rejected");

  close_request.fd = device_fd;
  failures += expect(functions->handle_drm_call(HSAKMT_DRM_CLOSE,
                                                 &close_request) == 0,
                     "device render descriptor closes through the model ABI");
  close_request.fd = render_fd;
  failures += expect(functions->handle_drm_call(HSAKMT_DRM_CLOSE,
                                                 &close_request) == 0,
                     "caller render descriptor closes through the model ABI");

  initialize_request.fd = STDOUT_FILENO;
  major = UINT32_C(0xa5a5a5a5);
  minor = UINT32_C(0xa5a5a5a5);
  device = (void *)(uintptr_t)UINT64_C(0xa5a5a5a5a5a5a5a5);
  errno = 0;
  failures += expect(functions->handle_drm_call(
                         HSAKMT_DRM_DEVICE_INITIALIZE,
                         &initialize_request) == -1 &&
                         errno == EBADF && major == UINT32_C(0xa5a5a5a5) &&
                         minor == UINT32_C(0xa5a5a5a5) &&
                         (uintptr_t)device == UINT64_C(0xa5a5a5a5a5a5a5a5),
                     "foreign descriptor is rejected before outputs commit");
  return failures;
}

int main(void) {
  get_hsakmt_model_functions_t getter;
  const struct hsakmt_model_functions *functions;
  struct stat state;
  struct kfd_ioctl_create_queue_args queue;
  struct kfd_ioctl_get_process_apertures_new_args apertures;
  struct kfd_ioctl_acquire_vm_args acquire_vm;
  struct kfd_ioctl_get_clock_counters_args clock;
  struct hsakmt_drm_bo_free_args drm_arguments;
  void *library;
  int descriptor;
  int failures = 0;

  library = dlopen(SAGR_HSAKMT_MODEL_PATH, RTLD_NOW | RTLD_LOCAL);
  failures += expect(library != NULL, "model DSO loads");
  if (library == NULL) {
    return 1;
  }
  getter = (get_hsakmt_model_functions_t)dlsym(
      library, "get_hsakmt_model_functions");
  failures += expect(getter != NULL, "official model getter is exported");
  if (getter == NULL) {
    (void)dlclose(library);
    return 1;
  }
  functions = getter();
  failures += expect(functions != NULL && functions->version_major == 1U &&
                         functions->version_minor == 1U &&
                         functions->create_memfd != NULL &&
                         functions->handle_ioctl != NULL &&
                         functions->handle_drm_call != NULL,
                     "model ABI 1.1 function table is complete");

  descriptor = functions->create_memfd();
  failures += expect(
      descriptor >= 0 && fstat(descriptor, &state) == 0 &&
          state.st_size == (off_t)MODEL_BACKING_BYTES && state.st_blocks >= 0 &&
          (uint64_t)state.st_blocks * UINT64_C(512) < MODEL_BACKING_BYTES &&
          (fcntl(descriptor, F_GETFD) & FD_CLOEXEC) != 0 &&
          (fcntl(descriptor, F_GET_SEALS) & (F_SEAL_SHRINK | F_SEAL_GROW)) ==
              (F_SEAL_SHRINK | F_SEAL_GROW),
      "model storage covers local memory sparsely and is size-sealed");
  failures += expect(functions->create_memfd() == descriptor,
                     "model storage ownership is stable within one process");
  if (descriptor >= 0) {
    failures += check_fork_storage(functions, state.st_ino);
  }
  failures += check_drm_lifecycle(functions);

  memset(&apertures, 0, sizeof(apertures));
  apertures.num_of_nodes = 1U;
  errno = 0;
  failures += expect(functions->handle_ioctl(
                         AMDKFD_IOC_GET_PROCESS_APERTURES_NEW,
                         &apertures) == -1 &&
                         errno == EINVAL && apertures.num_of_nodes == 1U,
                     "null aperture output fails before provider access");

  memset(&acquire_vm, 0, sizeof(acquire_vm));
  acquire_vm.drm_fd = STDOUT_FILENO;
  acquire_vm.gpu_id = 38144U;
  errno = 0;
  failures += expect(functions->handle_ioctl(
                         AMDKFD_IOC_ACQUIRE_VM, &acquire_vm) == -1 &&
                         errno == EBADF,
                     "VM acquisition rejects a foreign render descriptor");

  memset(&clock, 0xa5, sizeof(clock));
  clock.gpu_id = 0U;
  clock.pad = 0U;
  errno = 0;
  failures += expect(functions->handle_ioctl(
                         AMDKFD_IOC_GET_CLOCK_COUNTERS, &clock) == -1 &&
                         errno == EINVAL &&
                         clock.gpu_clock_counter ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5) &&
                         clock.cpu_clock_counter ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5) &&
                         clock.system_clock_counter ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5) &&
                         clock.system_clock_freq ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5),
                     "clock counters reject a foreign GPU atomically");

  memset(&clock, 0xa5, sizeof(clock));
  clock.gpu_id = 38144U;
  clock.pad = 1U;
  errno = 0;
  failures += expect(functions->handle_ioctl(
                         AMDKFD_IOC_GET_CLOCK_COUNTERS, &clock) == -1 &&
                         errno == EINVAL &&
                         clock.gpu_clock_counter ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5) &&
                         clock.cpu_clock_counter ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5) &&
                         clock.system_clock_counter ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5) &&
                         clock.system_clock_freq ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5),
                     "clock counters reject reserved input atomically");

  memset(&queue, 0xa5, sizeof(queue));
  errno = 0;
  failures += expect(functions->handle_ioctl(
                         AMDKFD_IOC_CREATE_QUEUE, &queue) == -1 &&
                         errno == EINVAL &&
                         queue.doorbell_offset ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5),
                     "invalid AQL queue carrier fails atomically");

  memset(&queue, 0, sizeof(queue));
  queue.gpu_id = 38144U;
  queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  queue.ring_base_address = UINT64_C(0x1000);
  queue.ring_size = 4096U;
  queue.read_pointer_address = UINT64_MAX - UINT64_C(7);
  queue.write_pointer_address = UINT64_C(0x2000);
  queue.queue_percentage = KFD_MAX_QUEUE_PERCENTAGE;
  queue.queue_priority = KFD_MAX_QUEUE_PRIORITY;
  queue.doorbell_offset = UINT64_C(0xa5a5a5a5a5a5a5a5);
  errno = 0;
  failures += expect(functions->handle_ioctl(
                         AMDKFD_IOC_CREATE_QUEUE, &queue) == -1 &&
                         errno == EINVAL &&
                         queue.doorbell_offset ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5),
                     "AQL read-index overflow fails before provider access");

  queue.read_pointer_address = UINT64_C(0x2000);
  queue.write_pointer_address = UINT64_MAX - UINT64_C(7);
  errno = 0;
  failures += expect(functions->handle_ioctl(
                         AMDKFD_IOC_CREATE_QUEUE, &queue) == -1 &&
                         errno == EINVAL &&
                         queue.doorbell_offset ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5),
                     "AQL write-index overflow fails before provider access");

  queue.write_pointer_address = UINT64_C(0x2008);
  queue.ring_base_address = UINT64_MAX - UINT64_C(63);
  errno = 0;
  failures += expect(functions->handle_ioctl(
                         AMDKFD_IOC_CREATE_QUEUE, &queue) == -1 &&
                         errno == EINVAL &&
                         queue.doorbell_offset ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5),
                     "AQL ring overflow fails before provider access");
  errno = 0;
  failures += expect(functions->handle_ioctl(0U, &queue) == -1 &&
                         errno == EINVAL,
                     "foreign ioctl is rejected before dispatch");

  memset(&drm_arguments, 0xa5, sizeof(drm_arguments));
  errno = 0;
  failures += expect(functions->handle_drm_call(
                         HSAKMT_DRM_BO_FREE, &drm_arguments) == -1 &&
                         errno == ENOTSUP &&
                         (uintptr_t)drm_arguments.bo ==
                             UINT64_C(0xa5a5a5a5a5a5a5a5),
                     "known unimplemented DRM command fails atomically");
  errno = 0;
  failures += expect(functions->handle_drm_call(
                         (unsigned)HSAKMT_DRM_QUERY_GPU_INFO + 1U,
                         &drm_arguments) == -1 &&
                         errno == EINVAL,
                     "foreign DRM command is rejected before dispatch");
  errno = 0;
  failures += expect(functions->handle_drm_call(
                         HSAKMT_DRM_BO_FREE, NULL) == -1 &&
                         errno == EINVAL,
                     "null DRM carrier is rejected");

  (void)dlclose(library);
  return failures == 0 ? 0 : 1;
}
