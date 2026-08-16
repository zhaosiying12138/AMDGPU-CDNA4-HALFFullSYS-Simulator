/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <hsakmt/hsakmtmodeliface.h>
#include <hsakmt/drm/amdgpu.h>
#include <hsakmt/linux/kfd_ioctl.h>

#include <self_amdgpu_runtime/kmt_shim.h>
#include <self_amdgpu_runtime/provider.h>

/*
 * This shared object implements the official libhsakmt model boundary. The
 * upstream libhsakmt remains responsible for its public HSAKMT surface and
 * translates those calls into KFD ioctls and the 1.1 DRM command set. This
 * provider never links host libhsakmt, libdrm, libdrm_amdgpu, or a KMD.
 *
 * Device semantics are enabled by capability family in this file. An ioctl or
 * DRM command without a typed runtime implementation fails without modifying
 * caller-owned output. This keeps upstream libhsakmt unchanged while the model
 * provider remains the only KMD-removal-specific boundary.
 */

#define SAGR_HSAKMT_MODEL_BACKING_PAGE_BYTES UINT64_C(4096)
#define SAGR_HSAKMT_MODEL_RENDER_FIRST 128
#define SAGR_HSAKMT_MODEL_RENDER_LAST 255
#define SAGR_HSAKMT_MODEL_DEVICE_CAPACITY 128U
#define SAGR_HSAKMT_MODEL_QUEUE_CAPACITY 128U
#define SAGR_HSAKMT_MODEL_AQL_PACKET_BYTES UINT64_C(64)
#define SAGR_HSAKMT_MODEL_DRM_MAJOR 3U
#define SAGR_HSAKMT_MODEL_DRM_MINOR 57U
#define SAGR_HSAKMT_MODEL_GFX950_DEVICE_ID 30112U
#define SAGR_HSAKMT_MODEL_GFX950_FAMILY_ID 160U
#define SAGR_HSAKMT_MODEL_VISIBLE_GPU_ID 38144U
#define SAGR_HSAKMT_MODEL_MAXIMUM_VISIBLE_GPUS 16U
#define SAGR_HSAKMT_MODEL_LOCAL_MEMORY_BYTES UINT64_C(309237645312)
#define SAGR_HSAKMT_MODEL_MEMFD_BYTES                                  \
  (SAGR_HSAKMT_MODEL_LOCAL_MEMORY_BYTES +                              \
   (uint64_t)SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES)

#define SAGR_HSAKMT_MODEL_ORDINARY_BACKING_BYTES                         \
  (SAGR_HSAKMT_MODEL_MEMFD_BYTES -                                      \
   (uint64_t)SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES)

_Static_assert(
    SAGR_HSAKMT_MODEL_MEMFD_BYTES >
        SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES,
    "the process backing must contain an ordinary allocation region");
_Static_assert(
    SAGR_HSAKMT_MODEL_ORDINARY_BACKING_BYTES ==
        SAGR_HSAKMT_MODEL_LOCAL_MEMORY_BYTES,
    "the sparse ordinary backing must cover the advertised local memory");
_Static_assert(sizeof(off_t) >= sizeof(int64_t),
               "the model backing requires 64-bit file offsets");
_Static_assert(sizeof(size_t) >= sizeof(uint64_t),
               "the model backing requires a 64-bit address space");
_Static_assert(
    SAGR_HSAKMT_MODEL_QUEUE_CAPACITY <=
        SAGR_BRIDGE_KMT_SHARED_BACKING_MAXIMUM_DOORBELL_SLOTS,
    "every model queue needs one generated doorbell slot");
_Static_assert(
    SAGR_BRIDGE_KMT_SHARED_BACKING_COMPLETION_REGION_BASE_BYTES +
            SAGR_HSAKMT_MODEL_QUEUE_CAPACITY *
                SAGR_BRIDGE_KMT_SHARED_BACKING_COMPLETION_SLOT_BYTES <=
        SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES,
    "every model queue needs one generated completion slot");
_Static_assert(
    KFD_IOC_ALLOC_MEM_FLAGS_USERPTR ==
        SAGR_BRIDGE_KMT_SHARED_BACKING_USERPTR_MEMORY_FLAG_MASK,
    "generated USERPTR mask must match the pinned upstream KFD ABI");
_Static_assert(
    KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL ==
        SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_MEMORY_FLAG_MASK,
    "generated DOORBELL mask must match the pinned upstream KFD ABI");

struct model_device {
  int active;
  pid_t owner_pid;
  int descriptor;
  int render_minor;
  uintptr_t token;
};

struct model_render {
  int active;
  pid_t owner_pid;
  int descriptor;
  int render_minor;
};

struct model_allocation {
  int active;
  int owns_backing;
  int mapped_by_model;
  pid_t owner_pid;
  uint64_t raw_handle;
  uint64_t va_addr;
  uint64_t size_bytes;
  uint64_t flags;
  uint64_t backing_offset;
  uint64_t backing_bytes;
  sagr_kmt_handle_t handle;
};

struct model_queue {
  int active;
  pid_t owner_pid;
  uint32_t raw_queue_id;
  uint32_t queue_type;
  uint64_t ring_base_address;
  uint64_t ring_size_bytes;
  uint64_t read_pointer_address;
  uint64_t write_pointer_address;
  uint64_t doorbell_offset;
  uint64_t last_doorbell_notification;
  uint64_t last_completion;
  size_t slot;
  uint32_t queue_percentage;
  uint32_t queue_priority;
  sagr_kmt_handle_t handle;
};

static pthread_mutex_t model_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t model_atfork_once = PTHREAD_ONCE_INIT;
static pid_t model_owner_pid;
static int model_memfd = -1;
static int model_atfork_error;
static int model_inherited_provider;
static sagr_provider_t *model_provider;
static sagr_kmt_handle_t model_kfd;
static int model_kfd_open;
static int model_backing_exported;
static struct model_device model_devices[SAGR_HSAKMT_MODEL_DEVICE_CAPACITY];
static struct model_render model_renders[SAGR_HSAKMT_MODEL_DEVICE_CAPACITY];
static struct model_allocation
    model_allocations[SAGR_HSAKMT_MODEL_DEVICE_CAPACITY * 8U];
static struct model_queue model_queues[SAGR_HSAKMT_MODEL_QUEUE_CAPACITY];
static uintptr_t model_next_device_token = (uintptr_t)1U;
static pthread_t model_progress_thread;
static int model_progress_started;
static int model_progress_stop;
static pid_t model_authorized_peer_pid = -1;

static const char model_marketing_name[] = "AMD Instinct MI350X";

static void model_reset_devices_locked(void) {
  size_t index;
  for (index = 0U; index < SAGR_HSAKMT_MODEL_DEVICE_CAPACITY; ++index) {
    if (model_devices[index].active != 0 && model_devices[index].descriptor >= 0) {
      (void)close(model_devices[index].descriptor);
    }
    if (model_renders[index].active != 0 && model_renders[index].descriptor >= 0) {
      (void)close(model_renders[index].descriptor);
    }
  }
  memset(model_devices, 0, sizeof(model_devices));
  memset(model_renders, 0, sizeof(model_renders));
  memset(model_allocations, 0, sizeof(model_allocations));
  memset(model_queues, 0, sizeof(model_queues));
}

static struct model_allocation *model_allocation_locked(uint64_t raw_handle) {
  size_t index;
  if (raw_handle == 0U) {
    return NULL;
  }
  for (index = 0U; index < sizeof(model_allocations) /
                                sizeof(model_allocations[0]); ++index) {
    if (model_allocations[index].active != 0 &&
        model_allocations[index].owner_pid == getpid() &&
        model_allocations[index].raw_handle == raw_handle) {
      return &model_allocations[index];
    }
  }
  return NULL;
}

static struct model_allocation *model_new_allocation_locked(void) {
  size_t index;
  for (index = 0U; index < sizeof(model_allocations) /
                                sizeof(model_allocations[0]); ++index) {
    if (model_allocations[index].active == 0) {
      return &model_allocations[index];
    }
  }
  return NULL;
}

static uint64_t model_available_vram_locked(void) {
  uint64_t available = SAGR_HSAKMT_MODEL_LOCAL_MEMORY_BYTES;
  size_t index;

  for (index = 0U; index < sizeof(model_allocations) /
                                sizeof(model_allocations[0]); ++index) {
    const struct model_allocation *allocation = &model_allocations[index];
    if (allocation->active == 0 || allocation->owner_pid != getpid() ||
        (allocation->flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) == 0U) {
      continue;
    }
    available = allocation->size_bytes >= available
                    ? 0U
                    : available - allocation->size_bytes;
  }
  return available;
}

static struct model_queue *model_queue_locked(uint32_t raw_queue_id) {
  size_t index;
  if (raw_queue_id == 0U) {
    return NULL;
  }
  for (index = 0U; index < SAGR_HSAKMT_MODEL_QUEUE_CAPACITY; ++index) {
    if (model_queues[index].active != 0 &&
        model_queues[index].owner_pid == getpid() &&
        model_queues[index].raw_queue_id == raw_queue_id) {
      return &model_queues[index];
    }
  }
  return NULL;
}

static struct model_queue *model_new_queue_locked(size_t *slot) {
  size_t index;
  if (slot == NULL) {
    return NULL;
  }
  for (index = 0U; index < SAGR_HSAKMT_MODEL_QUEUE_CAPACITY; ++index) {
    if (model_queues[index].active == 0) {
      *slot = index;
      return &model_queues[index];
    }
  }
  return NULL;
}

static int model_read_u64_at(uint64_t offset, uint64_t *value) {
  ssize_t result;
  if (model_memfd < 0 || value == NULL ||
      offset > SAGR_HSAKMT_MODEL_MEMFD_BYTES - sizeof(*value)) {
    return -1;
  }
  do {
    result = pread(model_memfd, value, sizeof(*value), (off_t)offset);
  } while (result < 0 && errno == EINTR);
  return result == (ssize_t)sizeof(*value) ? 0 : -1;
}

static int model_trace_enabled(void);

static void model_trace_queue_progress(const struct model_queue *queue,
                                       const char *phase, uint64_t doorbell,
                                       uint64_t notification,
                                       uint64_t completion, int status) {
  if (model_trace_enabled() == 0) {
    return;
  }
  flockfile(stderr);
  fprintf(stderr,
          "hsakmt-model pid=%ld phase=queue-%s queue_id=%u slot=%zu doorbell=%llu notification=%llu completion=%llu status=%d\n",
          (long)getpid(), phase, queue->raw_queue_id, queue->slot,
          (unsigned long long)doorbell,
          (unsigned long long)notification,
          (unsigned long long)completion, status);
  funlockfile(stderr);
}

static void *model_queue_progress_main(void *unused) {
  const struct timespec delay = {0, 1000000};
  (void)unused;
  for (;;) {
    size_t index;
    int lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      return NULL;
    }
    if (model_progress_stop != 0 || model_owner_pid != getpid()) {
      (void)pthread_mutex_unlock(&model_mutex);
      return NULL;
    }
    for (index = 0U; index < SAGR_HSAKMT_MODEL_QUEUE_CAPACITY; ++index) {
      struct model_queue *queue = &model_queues[index];
      uint64_t doorbell = 0;
      uint64_t completion = 0;
      uint64_t maximum_completion = 0;
      uint64_t notification = 0;
      const uint64_t completion_offset =
          SAGR_HSAKMT_MODEL_ORDINARY_BACKING_BYTES +
          SAGR_BRIDGE_KMT_SHARED_BACKING_COMPLETION_REGION_BASE_BYTES +
          (uint64_t)queue->slot *
              SAGR_BRIDGE_KMT_SHARED_BACKING_COMPLETION_SLOT_BYTES;
      if (queue->active == 0 || queue->owner_pid != getpid() ||
          queue->read_pointer_address == 0U ||
          (queue->read_pointer_address & UINT64_C(7)) != 0U ||
          model_read_u64_at(queue->doorbell_offset, &doorbell) != 0 ||
          model_read_u64_at(completion_offset, &completion) != 0 ||
          doorbell == SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_INITIAL_VALUE) {
        continue;
      }
      notification = doorbell + UINT64_C(1);
      if (notification < queue->last_doorbell_notification) {
        model_progress_stop = 1;
        (void)pthread_mutex_unlock(&model_mutex);
        return NULL;
      }
      if (notification > queue->last_doorbell_notification) {
        uint64_t accepted = 0;
        const sagr_kmt_status_t notify_status = sagr_kmt_queue_doorbell(
            model_provider, &model_kfd, &queue->handle, notification,
            &accepted, NULL, NULL, 0U);
        model_trace_queue_progress(queue, "doorbell", doorbell, notification,
                                   completion, (int)notify_status);
        if (notify_status != SAGR_KMT_STATUS_SUCCESS ||
            accepted != notification) {
          model_progress_stop = 1;
          (void)pthread_mutex_unlock(&model_mutex);
          return NULL;
        }
        queue->last_doorbell_notification = notification;
      }
      maximum_completion = doorbell + UINT64_C(1);
      if (completion <= queue->last_completion ||
          completion > maximum_completion) {
        continue;
      }
      __atomic_store_n(
          (uint64_t *)(uintptr_t)queue->read_pointer_address,
          completion, __ATOMIC_RELEASE);
      model_trace_queue_progress(queue, "retired", doorbell, notification,
                                 completion, 0);
      queue->last_completion = completion;
    }
    (void)pthread_mutex_unlock(&model_mutex);
    {
      struct timespec remaining = delay;
      while (nanosleep(&remaining, &remaining) != 0 && errno == EINTR) {
      }
    }
  }
}

static int model_start_queue_progress_locked(void) {
  int status;
  if (model_progress_started != 0) {
    return 0;
  }
  model_progress_stop = 0;
  status = pthread_create(
      &model_progress_thread, NULL, model_queue_progress_main, NULL);
  if (status != 0) {
    errno = status;
    return -1;
  }
  model_progress_started = 1;
  return 0;
}

static int model_backing_span(uint64_t size, uint64_t *span) {
  const uint64_t mask = SAGR_HSAKMT_MODEL_BACKING_PAGE_BYTES - UINT64_C(1);
  uint64_t rounded;
  if (span == NULL || size == 0U || size > UINT64_MAX - mask) {
    return -1;
  }
  rounded = (size + mask) & ~mask;
  if (rounded > SAGR_HSAKMT_MODEL_ORDINARY_BACKING_BYTES) {
    return -1;
  }
  *span = rounded;
  return 0;
}

static int model_backing_range_available_locked(uint64_t offset,
                                                uint64_t span) {
  uint64_t end;
  size_t index;
  if (span == 0U || offset > SAGR_HSAKMT_MODEL_MEMFD_BYTES - span) {
    return 0;
  }
  end = offset + span;
  for (index = 0U;
       index < sizeof(model_allocations) / sizeof(model_allocations[0]);
       ++index) {
    const struct model_allocation *allocation = &model_allocations[index];
    uint64_t allocation_end;
    if (allocation->active == 0 || allocation->owns_backing == 0) {
      continue;
    }
    if (allocation->backing_offset >
        UINT64_MAX - allocation->backing_bytes) {
      return 0;
    }
    allocation_end = allocation->backing_offset + allocation->backing_bytes;
    if (offset < allocation_end && allocation->backing_offset < end) {
      return 0;
    }
  }
  return 1;
}

/* The model owns the memfd, so it also owns this file-offset namespace. */
static int model_allocate_backing_locked(uint64_t size, uint64_t *offset,
                                         uint64_t *span) {
  uint64_t candidate = 0;
  uint64_t required = 0;
  size_t index;
  if (offset == NULL || model_backing_span(size, &required) != 0) {
    return -1;
  }
  for (;;) {
    int advanced = 0;
    if (candidate > SAGR_HSAKMT_MODEL_ORDINARY_BACKING_BYTES - required) {
      return -1;
    }
    for (index = 0U; index < sizeof(model_allocations) /
                                  sizeof(model_allocations[0]); ++index) {
      const struct model_allocation *allocation = &model_allocations[index];
      const uint64_t candidate_end = candidate + required;
      uint64_t allocation_end;
      if (allocation->active == 0 || allocation->owns_backing == 0) {
        continue;
      }
      allocation_end = allocation->backing_offset + allocation->backing_bytes;
      if (candidate < allocation_end &&
          allocation->backing_offset < candidate_end) {
        candidate = allocation_end;
        advanced = 1;
        break;
      }
    }
    if (advanced == 0) {
      *offset = candidate;
      if (span != NULL) {
        *span = required;
      }
      return 0;
    }
  }
}

static void model_atfork_prepare(void) { (void)pthread_mutex_lock(&model_mutex); }

static void model_atfork_parent(void) { (void)pthread_mutex_unlock(&model_mutex); }

static void model_atfork_child(void) {
  model_reset_devices_locked();
  if (model_memfd >= 0) {
    (void)close(model_memfd);
  }
  model_memfd = -1;
  /* A provider connection is process-owned. The child must never share its
   * request stream or terminate a parent-owned private gem5 process. It drops
   * the inherited local copy and reconnects lazily on its first model call. */
  model_inherited_provider = model_provider != NULL ? 1 : 0;
  model_kfd_open = 0;
  model_backing_exported = 0;
  model_progress_started = 0;
  model_progress_stop = 0;
#if defined(__linux__) && defined(PR_SET_PTRACER)
  (void)prctl(PR_SET_PTRACER, 0UL, 0UL, 0UL, 0UL);
#endif
  model_authorized_peer_pid = -1;
  model_owner_pid = getpid();
  (void)pthread_mutex_unlock(&model_mutex);
}

static void model_install_atfork(void) {
  model_owner_pid = getpid();
  model_atfork_error = pthread_atfork(model_atfork_prepare,
                                      model_atfork_parent,
                                      model_atfork_child);
}

static int model_create_storage(void) {
  int descriptor;
  int flags = MFD_CLOEXEC;
#ifdef MFD_ALLOW_SEALING
  flags |= MFD_ALLOW_SEALING;
#endif
  descriptor = memfd_create("self-amdgpu-hsakmt-model", (unsigned int)flags);
  if (descriptor < 0) {
    return -1;
  }
  if (ftruncate(descriptor, (off_t)SAGR_HSAKMT_MODEL_MEMFD_BYTES) != 0) {
    const int saved_errno = errno;
    (void)close(descriptor);
    errno = saved_errno;
    return -1;
  }
#ifdef F_ADD_SEALS
  if (fcntl(descriptor, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) != 0) {
    const int saved_errno = errno;
    (void)close(descriptor);
    errno = saved_errno;
    return -1;
  }
#endif
  return descriptor;
}

static int model_descriptor_is_storage_locked(int descriptor) {
  struct stat candidate;
  struct stat storage;
  if (descriptor < 0 || model_memfd < 0 || fstat(descriptor, &candidate) != 0 ||
      fstat(model_memfd, &storage) != 0) {
    return 0;
  }
  return candidate.st_dev == storage.st_dev && candidate.st_ino == storage.st_ino;
}

static struct model_device *model_device_locked(void *handle) {
  const uintptr_t token = (uintptr_t)handle;
  size_t index;
  if (token == (uintptr_t)0U) {
    return NULL;
  }
  for (index = 0U; index < SAGR_HSAKMT_MODEL_DEVICE_CAPACITY; ++index) {
    if (model_devices[index].active != 0 &&
        model_devices[index].token == token &&
        model_devices[index].owner_pid == getpid() &&
        model_devices[index].descriptor >= 0) {
      return &model_devices[index];
    }
  }
  return NULL;
}

static struct model_device *model_allocate_device_locked(void) {
  size_t index;
  for (index = 0U; index < SAGR_HSAKMT_MODEL_DEVICE_CAPACITY; ++index) {
    if (model_devices[index].active == 0) {
      return &model_devices[index];
    }
  }
  return NULL;
}

static struct model_render *model_render_locked(int descriptor) {
  size_t index;
  for (index = 0U; index < SAGR_HSAKMT_MODEL_DEVICE_CAPACITY; ++index) {
    if (model_renders[index].active != 0 &&
        model_renders[index].owner_pid == getpid() &&
        model_renders[index].descriptor == descriptor &&
        model_descriptor_is_storage_locked(descriptor)) {
      return &model_renders[index];
    }
  }
  return NULL;
}

static struct model_render *model_allocate_render_locked(int descriptor) {
  struct model_render *available = NULL;
  size_t index;
  for (index = 0U; index < SAGR_HSAKMT_MODEL_DEVICE_CAPACITY; ++index) {
    if (model_renders[index].active != 0 &&
        (model_renders[index].descriptor == descriptor ||
         !model_descriptor_is_storage_locked(model_renders[index].descriptor))) {
      memset(&model_renders[index], 0, sizeof(model_renders[index]));
    } else if (available == NULL && model_renders[index].active == 0) {
      available = &model_renders[index];
    }
  }
  return available;
}

static int model_create_memfd(void) {
  int descriptor;
  if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
      model_atfork_error != 0) {
    errno = model_atfork_error != 0 ? model_atfork_error : EIO;
    return -1;
  }
  if (pthread_mutex_lock(&model_mutex) != 0) {
    errno = EDEADLK;
    return -1;
  }
  if (model_owner_pid != getpid()) {
    if (model_memfd >= 0) {
      (void)close(model_memfd);
    }
    model_memfd = -1;
    model_owner_pid = getpid();
  }
  if (model_inherited_provider != 0) {
    (void)sagr_provider_discard_inherited(&model_provider);
    model_inherited_provider = 0;
  }
  if (model_memfd < 0) {
    model_memfd = model_create_storage();
  }
  descriptor = model_memfd;
  (void)pthread_mutex_unlock(&model_mutex);
  return descriptor;
}

static int model_runtime_status_errno(sagr_status_t status) {
  switch (status) {
    case SAGR_STATUS_INVALID_ARGUMENT:
      return EINVAL;
    case SAGR_STATUS_INVALID_HANDLE:
    case SAGR_STATUS_INSTANCE_MISMATCH:
      return EBADF;
    case SAGR_STATUS_NOT_SUPPORTED:
    case SAGR_STATUS_CAPABILITY_MISMATCH:
      return ENOTSUP;
    case SAGR_STATUS_ENDPOINT_NOT_FOUND:
    case SAGR_STATUS_UNAVAILABLE:
      return ENODEV;
    case SAGR_STATUS_TIMED_OUT:
      return ETIMEDOUT;
    case SAGR_STATUS_OUT_OF_RESOURCES:
      return ENOMEM;
    case SAGR_STATUS_BUSY:
      return EBUSY;
    case SAGR_STATUS_UNAUTHORIZED:
      return EACCES;
    case SAGR_STATUS_CONNECTION_LOST:
    case SAGR_STATUS_PROTOCOL_ERROR:
    case SAGR_STATUS_CHECKSUM_ERROR:
    case SAGR_STATUS_INTERNAL_ERROR:
    case SAGR_STATUS_VERSION_MISMATCH:
    case SAGR_STATUS_TOPOLOGY_MISMATCH:
    case SAGR_STATUS_BUFFER_TOO_SMALL:
    case SAGR_STATUS_CANCELLED:
    default:
      return EIO;
  }
}

static int model_kmt_status_errno(sagr_kmt_status_t status) {
  switch (status) {
    case SAGR_KMT_STATUS_INVALID_PARAMETER:
      return EINVAL;
    case SAGR_KMT_STATUS_INVALID_HANDLE:
    case SAGR_KMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED:
      return EBADF;
    case SAGR_KMT_STATUS_NOT_IMPLEMENTED:
    case SAGR_KMT_STATUS_NOT_SUPPORTED:
      return ENOTSUP;
    case SAGR_KMT_STATUS_UNAVAILABLE:
      return ENODEV;
    case SAGR_KMT_STATUS_NO_MEMORY:
    case SAGR_KMT_STATUS_OUT_OF_RESOURCES:
      return ENOMEM;
    case SAGR_KMT_STATUS_WAIT_TIMEOUT:
      return ETIMEDOUT;
    case SAGR_KMT_STATUS_BUFFER_TOO_SMALL:
    case SAGR_KMT_STATUS_INVALID_NODE_UNIT:
    case SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR:
    case SAGR_KMT_STATUS_WAIT_FAILURE:
    case SAGR_KMT_STATUS_MEMORY_ALIGNMENT:
    case SAGR_KMT_STATUS_ERROR:
    default:
      return EIO;
  }
}

static void model_revoke_peer_memory_access_locked(void) {
#if defined(__linux__) && defined(PR_SET_PTRACER)
  if (model_authorized_peer_pid > 0) {
    (void)prctl(PR_SET_PTRACER, 0UL, 0UL, 0UL, 0UL);
  }
#endif
  model_authorized_peer_pid = -1;
}

static int model_authorize_peer_memory_access_locked(void) {
  sagr_provider_info_t info;
#if !defined(__linux__) || !defined(PR_SET_PTRACER)
  errno = ENOTSUP;
  return -1;
#else
  if (model_provider == NULL) {
    errno = EBADF;
    return -1;
  }
  memset(&info, 0, sizeof(info));
  if (sagr_provider_get_info(model_provider, &info, (uint32_t)sizeof(info)) !=
          SAGR_STATUS_SUCCESS ||
      info.state != SAGR_PROVIDER_STATE_OPEN || info.peer_pid == 0U ||
      info.peer_pid > (uint32_t)INT32_MAX ||
      info.peer_uid != (uint32_t)geteuid() ||
      info.peer_pid == (uint32_t)getpid()) {
    errno = EACCES;
    return -1;
  }
  if (model_authorized_peer_pid == (pid_t)info.peer_pid) {
    return 0;
  }
  if (prctl(PR_SET_PTRACER, (unsigned long)info.peer_pid,
            0UL, 0UL, 0UL) != 0) {
    return -1;
  }
  model_authorized_peer_pid = (pid_t)info.peer_pid;
  return 0;
#endif
}

static int model_connect_locked(void) {
  sagr_error_info_t error;
  sagr_status_t runtime_status;
  sagr_kmt_status_t kmt_status;
  if (model_inherited_provider != 0) {
    const sagr_status_t discard_status =
        sagr_provider_discard_inherited(&model_provider);
    if (discard_status != SAGR_STATUS_SUCCESS) {
      errno = model_runtime_status_errno(discard_status);
      return -1;
    }
    model_inherited_provider = 0;
  }
  if (model_provider != NULL && model_kfd_open != 0 &&
      model_backing_exported != 0 && model_authorized_peer_pid > 0) {
    return 0;
  }
  if (model_memfd < 0) {
    model_memfd = model_create_storage();
    if (model_memfd < 0) {
      return -1;
    }
  }
  memset(&error, 0, sizeof(error));
  runtime_status = sagr_provider_open_managed(
      NULL, &model_provider, NULL, 0U, &error, (uint32_t)sizeof(error));
  if (runtime_status != SAGR_STATUS_SUCCESS) {
    model_provider = NULL;
    errno = model_runtime_status_errno(runtime_status);
    return -1;
  }
  if (model_authorize_peer_memory_access_locked() != 0) {
    const int saved_errno = errno != 0 ? errno : EACCES;
    (void)sagr_provider_close(&model_provider);
    model_authorized_peer_pid = -1;
    errno = saved_errno;
    return -1;
  }
  memset(&model_kfd, 0, sizeof(model_kfd));
  kmt_status = sagr_kmt_open_kfd(model_provider, &model_kfd, NULL, &error,
                                 (uint32_t)sizeof(error));
  if (kmt_status != SAGR_KMT_STATUS_SUCCESS) {
    (void)sagr_provider_close(&model_provider);
    memset(&model_kfd, 0, sizeof(model_kfd));
    model_revoke_peer_memory_access_locked();
    errno = model_kmt_status_errno(kmt_status);
    return -1;
  }
  model_kfd_open = 1;
  kmt_status = sagr_kmt_export_backing(
      model_provider, &model_kfd, model_memfd,
      SAGR_HSAKMT_MODEL_MEMFD_BYTES,
      (uint32_t)SAGR_HSAKMT_MODEL_BACKING_PAGE_BYTES, NULL, &error,
      (uint32_t)sizeof(error));
  if (kmt_status != SAGR_KMT_STATUS_SUCCESS) {
    (void)sagr_kmt_close_kfd(model_provider, &model_kfd, NULL, NULL, 0U);
    model_kfd_open = 0;
    model_backing_exported = 0;
    memset(&model_kfd, 0, sizeof(model_kfd));
    (void)sagr_provider_close(&model_provider);
    model_revoke_peer_memory_access_locked();
    errno = model_kmt_status_errno(kmt_status);
    return -1;
  }
  model_backing_exported = 1;
  return 0;
}

static void model_disconnect_locked(void) {
  if (model_inherited_provider != 0) {
    (void)sagr_provider_discard_inherited(&model_provider);
    model_inherited_provider = 0;
    return;
  }
  if (model_provider != NULL && model_kfd_open != 0) {
    (void)sagr_kmt_close_kfd(model_provider, &model_kfd, NULL, NULL, 0U);
  }
  model_kfd_open = 0;
  model_backing_exported = 0;
  memset(&model_kfd, 0, sizeof(model_kfd));
  if (model_provider != NULL) {
    (void)sagr_provider_close(&model_provider);
  }
  model_revoke_peer_memory_access_locked();
}

static int model_handle_ioctl_impl(unsigned long request, void *argument) {
  int lock_status;
  if (_IOC_TYPE(request) != AMDKFD_IOCTL_BASE || argument == NULL) {
    errno = EINVAL;
    return -1;
  }
  if (request == AMDKFD_IOC_GET_VERSION) {
    struct kfd_ioctl_get_version_args committed;
    sagr_kmt_version_t version;
    sagr_kmt_status_t status;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    if (model_owner_pid != getpid() || model_connect_locked() != 0) {
      const int saved_errno = errno != 0 ? errno : EIO;
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    memset(&version, 0, sizeof(version));
    status = sagr_kmt_get_version(
        model_provider, &model_kfd, &version, (uint32_t)sizeof(version), NULL,
        NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS) {
      const int saved_errno = model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    memset(&committed, 0, sizeof(committed));
    committed.major_version = version.major;
    committed.minor_version = version.minor;
    memcpy(argument, &committed, sizeof(committed));
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_GET_CLOCK_COUNTERS) {
    const struct kfd_ioctl_get_clock_counters_args *ioctl_args = argument;
    struct kfd_ioctl_get_clock_counters_args committed;
    sagr_kmt_clock_counters_t counters;
    sagr_kmt_status_t status;
    if (ioctl_args->gpu_id != SAGR_HSAKMT_MODEL_VISIBLE_GPU_ID ||
        ioctl_args->pad != 0U) {
      errno = EINVAL;
      return -1;
    }
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    if (model_owner_pid != getpid() || model_connect_locked() != 0) {
      const int saved_errno =
          model_owner_pid != getpid() ? EOWNERDEAD : (errno != 0 ? errno : EIO);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    memset(&counters, 0, sizeof(counters));
    status = sagr_kmt_get_clock_counters(
        model_provider, &model_kfd, ioctl_args->gpu_id, &counters,
        (uint32_t)sizeof(counters), NULL, NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS) {
      const int saved_errno = model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    memset(&committed, 0, sizeof(committed));
    committed.gpu_clock_counter = counters.gpu_clock_counter;
    committed.cpu_clock_counter = counters.cpu_clock_counter;
    committed.system_clock_counter = counters.system_clock_counter;
    committed.system_clock_freq = counters.system_clock_frequency_hz;
    committed.gpu_id = ioctl_args->gpu_id;
    memcpy(argument, &committed, sizeof(committed));
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_GET_PROCESS_APERTURES_NEW) {
    struct kfd_ioctl_get_process_apertures_new_args *ioctl_args = argument;
    sagr_kmt_process_aperture_t committed[SAGR_HSAKMT_MODEL_MAXIMUM_VISIBLE_GPUS];
    struct kfd_process_device_apertures *destination =
        (struct kfd_process_device_apertures *)(uintptr_t)
            ioctl_args->kfd_process_device_apertures_ptr;
    uint32_t requested = ioctl_args->num_of_nodes;
    uint32_t total = 0U;
    uint32_t expected_total = 0U;
    uint32_t returned = 0U;
    uint32_t written = 0U;
    sagr_kmt_status_t status;
    if (requested != 0U && destination == NULL) {
      errno = EINVAL;
      return -1;
    }
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    if (model_owner_pid != getpid() || model_connect_locked() != 0) {
      const int saved_errno = errno != 0 ? errno : EIO;
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    status = sagr_kmt_process_apertures(
        model_provider, &model_kfd, 0U, NULL, 0U, &returned, &total, NULL,
        NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS || returned != 0U ||
        total == 0U || total > SAGR_HSAKMT_MODEL_MAXIMUM_VISIBLE_GPUS) {
      const int saved_errno = status == SAGR_KMT_STATUS_SUCCESS
                                  ? EPROTO
                                  : model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    expected_total = total;
    while (written < requested && written < total) {
      uint32_t page_total = 0U;
      const uint32_t capacity =
          requested - written < SAGR_KMT_PROCESS_APERTURES_PER_PAGE
              ? requested - written
              : SAGR_KMT_PROCESS_APERTURES_PER_PAGE;
      status = sagr_kmt_process_apertures(
          model_provider, &model_kfd, written, committed + written, capacity,
          &returned, &page_total, NULL, NULL, 0U);
      if (status != SAGR_KMT_STATUS_SUCCESS || returned == 0U ||
          page_total != expected_total) {
        const int saved_errno = status == SAGR_KMT_STATUS_SUCCESS
                                    ? EPROTO
                                    : model_kmt_status_errno(status);
        (void)pthread_mutex_unlock(&model_mutex);
        errno = saved_errno;
        return -1;
      }
      written += returned;
    }
    if (destination != NULL) {
      uint32_t index;
      for (index = 0U; index < written; ++index) {
        struct kfd_process_device_apertures entry;
        memset(&entry, 0, sizeof(entry));
        entry.lds_base = committed[index].lds_base;
        entry.lds_limit = committed[index].lds_limit;
        entry.scratch_base = committed[index].scratch_base;
        entry.scratch_limit = committed[index].scratch_limit;
        entry.gpuvm_base = committed[index].gpuvm_base;
        entry.gpuvm_limit = committed[index].gpuvm_limit;
        entry.gpu_id = committed[index].gpu_id;
        memcpy(destination + index, &entry, sizeof(entry));
      }
    }
    ioctl_args->num_of_nodes = destination == NULL ? expected_total : written;
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_ACQUIRE_VM) {
    const struct kfd_ioctl_acquire_vm_args *ioctl_args = argument;
    struct model_render *render;
    sagr_kmt_status_t status;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    if (model_owner_pid != getpid()) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = EOWNERDEAD;
      return -1;
    }
    render = model_render_locked((int)ioctl_args->drm_fd);
    if (render == NULL) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = EBADF;
      return -1;
    }
    if (ioctl_args->gpu_id != SAGR_HSAKMT_MODEL_VISIBLE_GPU_ID) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = EINVAL;
      return -1;
    }
    if (model_connect_locked() != 0) {
      const int saved_errno = errno != 0 ? errno : EIO;
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    status = sagr_kmt_acquire_vm(
        model_provider, &model_kfd, ioctl_args->gpu_id,
        (uint32_t)render->render_minor, NULL, NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS) {
      const int saved_errno = model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_SET_MEMORY_POLICY) {
    const struct kfd_ioctl_set_memory_policy_args *ioctl_args = argument;
    sagr_kmt_status_t status;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    if (model_owner_pid != getpid()) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = EOWNERDEAD;
      return -1;
    }
    if (ioctl_args->gpu_id != SAGR_HSAKMT_MODEL_VISIBLE_GPU_ID ||
        (ioctl_args->default_policy != SAGR_KMT_CACHE_POLICY_COHERENT &&
         ioctl_args->default_policy != SAGR_KMT_CACHE_POLICY_NONCOHERENT) ||
        (ioctl_args->alternate_policy != SAGR_KMT_CACHE_POLICY_COHERENT &&
         ioctl_args->alternate_policy != SAGR_KMT_CACHE_POLICY_NONCOHERENT) ||
        ioctl_args->alternate_aperture_size == 0U ||
        (ioctl_args->alternate_aperture_base & UINT64_C(0xffff)) != 0U ||
        (ioctl_args->alternate_aperture_size & UINT64_C(0xffff)) != 0U ||
        ioctl_args->alternate_aperture_base >
            UINT64_MAX - ioctl_args->alternate_aperture_size) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = EINVAL;
      return -1;
    }
    if (model_connect_locked() != 0) {
      const int saved_errno = errno != 0 ? errno : EIO;
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    status = sagr_kmt_set_memory_policy(
        model_provider, &model_kfd, ioctl_args->gpu_id,
        ioctl_args->default_policy, ioctl_args->alternate_policy,
        ioctl_args->misc_process_flag, ioctl_args->alternate_aperture_base,
        ioctl_args->alternate_aperture_size, NULL, NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS) {
      const int saved_errno = model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_AVAILABLE_MEMORY) {
    struct kfd_ioctl_get_available_memory_args *ioctl_args = argument;
    uint64_t available;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    if (model_owner_pid != getpid() ||
        ioctl_args->gpu_id != SAGR_HSAKMT_MODEL_VISIBLE_GPU_ID) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = model_owner_pid != getpid() ? EOWNERDEAD : EINVAL;
      return -1;
    }
    available = model_available_vram_locked();
    ioctl_args->available = available;
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
    struct kfd_ioctl_alloc_memory_of_gpu_args *ioctl_args = argument;
    sagr_kmt_handle_t allocation;
    struct model_allocation *local;
    const int is_userptr =
        (ioctl_args->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) != 0U;
    const int is_doorbell =
        (ioctl_args->flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) != 0U;
    uint64_t backing_offset = 0;
    uint64_t backing_bytes = 0;
    uint64_t transport_offset = 0;
    uint64_t mmap_offset = 0;
    sagr_kmt_status_t status;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    if (model_owner_pid != getpid() ||
        ioctl_args->gpu_id != SAGR_HSAKMT_MODEL_VISIBLE_GPU_ID ||
        ioctl_args->size == 0U ||
        ioctl_args->va_addr > UINT64_MAX - ioctl_args->size ||
        (is_userptr != 0 &&
         ioctl_args->mmap_offset > UINT64_MAX - ioctl_args->size) ||
        (is_userptr != 0 && is_doorbell != 0) ||
        (is_doorbell != 0 &&
         ioctl_args->size !=
             SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES) ||
        (is_userptr != 0 && ioctl_args->mmap_offset == 0U) ||
        (is_userptr == 0 && ioctl_args->mmap_offset != 0U)) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = model_owner_pid != getpid() ? EOWNERDEAD : EINVAL;
      return -1;
    }
    if ((ioctl_args->flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) != 0U &&
        ioctl_args->size > model_available_vram_locked()) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = ENOMEM;
      return -1;
    }
    if (model_connect_locked() != 0) {
      const int saved_errno = errno != 0 ? errno : EIO;
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    local = model_new_allocation_locked();
    if (local == NULL) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = ENOMEM;
      return -1;
    }
    if (is_userptr != 0) {
      transport_offset = ioctl_args->mmap_offset;
    } else if (is_doorbell != 0) {
      backing_offset = SAGR_HSAKMT_MODEL_ORDINARY_BACKING_BYTES;
      backing_bytes =
          SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES;
      if (!model_backing_range_available_locked(backing_offset,
                                                backing_bytes)) {
        (void)pthread_mutex_unlock(&model_mutex);
        errno = ENOMEM;
        return -1;
      }
      transport_offset = backing_offset;
    } else if (model_allocate_backing_locked(ioctl_args->size, &backing_offset,
                                             &backing_bytes) != 0) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = ENOMEM;
      return -1;
    } else {
      transport_offset = backing_offset;
    }
    status = sagr_kmt_alloc_memory_of_gpu(
        model_provider, &model_kfd, ioctl_args->va_addr, ioctl_args->size,
        ioctl_args->gpu_id, ioctl_args->flags, transport_offset,
        &allocation, &mmap_offset, NULL, NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS) {
      const int saved_errno = model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    if (mmap_offset != transport_offset) {
      if (sagr_kmt_free_memory_of_gpu(model_provider, &model_kfd, &allocation,
                                      NULL, NULL, 0U) !=
          SAGR_KMT_STATUS_SUCCESS) {
        model_disconnect_locked();
      }
      (void)pthread_mutex_unlock(&model_mutex);
      errno = EPROTO;
      return -1;
    }
    memset(local, 0, sizeof(*local));
    local->active = 1;
    local->owns_backing = is_userptr == 0 ? 1 : 0;
    local->mapped_by_model = 0;
    local->owner_pid = getpid();
    local->raw_handle = allocation.object_id;
    local->va_addr = ioctl_args->va_addr;
    local->size_bytes = ioctl_args->size;
    local->flags = ioctl_args->flags;
    local->backing_offset = backing_offset;
    local->backing_bytes = backing_bytes;
    local->handle = allocation;
    /* The upstream FFM deliberately skips its normal CPU mmap for MMIO
     * remaps while model mode is active.  Publish the provider-owned sparse
     * backing at the exact aperture address so the standard ROCr doorbell
     * path can store to it. */
    if ((ioctl_args->flags & KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP) != 0U) {
      void *mapped;
      if (is_userptr != 0 || model_memfd < 0 ||
          (ioctl_args->va_addr & (SAGR_HSAKMT_MODEL_BACKING_PAGE_BYTES - 1U)) !=
              0U || (backing_offset & (SAGR_HSAKMT_MODEL_BACKING_PAGE_BYTES - 1U)) !=
              0U) {
        (void)sagr_kmt_free_memory_of_gpu(model_provider, &model_kfd,
                                          &allocation, NULL, NULL, 0U);
        memset(local, 0, sizeof(*local));
        (void)pthread_mutex_unlock(&model_mutex);
        errno = EINVAL;
        return -1;
      }
      mapped = mmap((void *)(uintptr_t)ioctl_args->va_addr,
                    (size_t)ioctl_args->size, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED, model_memfd,
                    (off_t)backing_offset);
      if (mapped == MAP_FAILED) {
        const int saved_errno = errno != 0 ? errno : ENOMEM;
        (void)sagr_kmt_free_memory_of_gpu(model_provider, &model_kfd,
                                          &allocation, NULL, NULL, 0U);
        memset(local, 0, sizeof(*local));
        (void)pthread_mutex_unlock(&model_mutex);
        errno = saved_errno;
        return -1;
      }
      (void)madvise(mapped, (size_t)ioctl_args->size, MADV_DONTFORK);
      local->mapped_by_model = 1;
    }
    ioctl_args->handle = allocation.object_id;
    ioctl_args->mmap_offset = mmap_offset;
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_FREE_MEMORY_OF_GPU) {
    const struct kfd_ioctl_free_memory_of_gpu_args *ioctl_args = argument;
    struct model_allocation *local;
    sagr_kmt_status_t status;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    local = model_allocation_locked(ioctl_args->handle);
    if (local == NULL || model_owner_pid != getpid() ||
        model_connect_locked() != 0) {
      const int saved_errno = model_owner_pid != getpid()
                                  ? EOWNERDEAD
                                  : (errno != 0 ? errno : EBADF);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    status = sagr_kmt_free_memory_of_gpu(model_provider, &model_kfd,
                                         &local->handle, NULL, NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS) {
      const int saved_errno = model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    if (local->mapped_by_model != 0 && local->va_addr != 0U &&
        local->size_bytes != 0U) {
      (void)munmap((void *)(uintptr_t)local->va_addr,
                   (size_t)local->size_bytes);
    }
    memset(local, 0, sizeof(*local));
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_MAP_MEMORY_TO_GPU ||
      request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU) {
    const struct kfd_ioctl_map_memory_to_gpu_args *ioctl_args = argument;
    uint32_t gpu_ids[SAGR_HSAKMT_MODEL_MAXIMUM_VISIBLE_GPUS];
    struct model_allocation *local;
    uint32_t success = 0U;
    sagr_kmt_status_t status;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    if (ioctl_args->n_devices == 0U ||
        ioctl_args->n_devices > SAGR_HSAKMT_MODEL_MAXIMUM_VISIBLE_GPUS ||
        ioctl_args->device_ids_array_ptr == 0U) {
      errno = EINVAL;
      return -1;
    }
    memcpy(gpu_ids, (const void *)(uintptr_t)ioctl_args->device_ids_array_ptr,
           (size_t)ioctl_args->n_devices * sizeof(gpu_ids[0]));
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    local = model_allocation_locked(ioctl_args->handle);
    if (local == NULL || model_owner_pid != getpid() ||
        model_connect_locked() != 0) {
      const int saved_errno = model_owner_pid != getpid()
                                  ? EOWNERDEAD
                                  : (errno != 0 ? errno : EBADF);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    status = request == AMDKFD_IOC_MAP_MEMORY_TO_GPU
        ? sagr_kmt_map_memory_to_gpu(model_provider, &model_kfd,
                                     &local->handle, gpu_ids,
                                     ioctl_args->n_devices, &success, NULL,
                                     NULL, 0U)
        : sagr_kmt_unmap_memory_from_gpu(model_provider, &model_kfd,
                                         &local->handle, gpu_ids,
                                         ioctl_args->n_devices, &success, NULL,
                                         NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS) {
      const int saved_errno = model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    ((struct kfd_ioctl_map_memory_to_gpu_args *)ioctl_args)->n_success =
        success;
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_CREATE_QUEUE) {
    struct kfd_ioctl_create_queue_args *ioctl_args = argument;
    sagr_kmt_queue_options_t options;
    sagr_kmt_handle_t queue_handle;
    struct model_queue *local;
    size_t slot = 0U;
    sagr_kmt_status_t status;
    const uint64_t packet_count =
        (uint64_t)ioctl_args->ring_size / SAGR_HSAKMT_MODEL_AQL_PACKET_BYTES;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    if (model_owner_pid != getpid() ||
        ioctl_args->gpu_id != SAGR_HSAKMT_MODEL_VISIBLE_GPU_ID ||
        ioctl_args->queue_type != KFD_IOC_QUEUE_TYPE_COMPUTE_AQL ||
        ioctl_args->ring_base_address == 0U ||
        ioctl_args->read_pointer_address == 0U ||
        ioctl_args->write_pointer_address == 0U ||
        (ioctl_args->ring_base_address & UINT64_C(63)) != 0U ||
        (ioctl_args->read_pointer_address & UINT64_C(7)) != 0U ||
        (ioctl_args->write_pointer_address & UINT64_C(7)) != 0U ||
        ioctl_args->ring_size < UINT32_C(4096) ||
        (ioctl_args->ring_size & (ioctl_args->ring_size - 1U)) != 0U ||
        (ioctl_args->ring_size % SAGR_HSAKMT_MODEL_AQL_PACKET_BYTES) != 0U ||
        ioctl_args->ring_base_address >
            UINT64_MAX - (uint64_t)ioctl_args->ring_size ||
        ioctl_args->read_pointer_address > UINT64_MAX - sizeof(uint64_t) ||
        ioctl_args->write_pointer_address > UINT64_MAX - sizeof(uint64_t) ||
        packet_count > (uint64_t)SAGR_KMT_QUEUE_MAX_DEPTH ||
        packet_count == 0U ||
        ioctl_args->queue_percentage > KFD_MAX_QUEUE_PERCENTAGE ||
        ioctl_args->queue_priority > KFD_MAX_QUEUE_PRIORITY) {
      errno = model_owner_pid != getpid() ? EOWNERDEAD : EINVAL;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    local = model_new_queue_locked(&slot);
    if (local == NULL || model_connect_locked() != 0) {
      const int saved_errno = local == NULL
                                  ? ENOMEM
                                  : (errno != 0 ? errno : EIO);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    memset(&options, 0, sizeof(options));
    options.struct_size = (uint32_t)sizeof(options);
    options.node_id = 1U;
    options.queue_type = ioctl_args->queue_type;
    options.depth = (uint32_t)packet_count;
    options.ring_base_address = ioctl_args->ring_base_address;
    options.ring_size_bytes = ioctl_args->ring_size;
    options.read_pointer_address = ioctl_args->read_pointer_address;
    options.write_pointer_address = ioctl_args->write_pointer_address;
    status = sagr_kmt_queue_create(model_provider, &model_kfd, &options,
                                    &queue_handle, NULL, NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS || queue_handle.object_id == 0U) {
      const int saved_errno = status == SAGR_KMT_STATUS_SUCCESS
                                  ? EPROTO
                                  : model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    memset(local, 0, sizeof(*local));
    local->active = 1;
    local->owner_pid = getpid();
    local->raw_queue_id = (uint32_t)(slot + 1U);
    local->queue_type = ioctl_args->queue_type;
    local->ring_base_address = ioctl_args->ring_base_address;
    local->ring_size_bytes = ioctl_args->ring_size;
    local->read_pointer_address = ioctl_args->read_pointer_address;
    local->write_pointer_address = ioctl_args->write_pointer_address;
    local->doorbell_offset =
        SAGR_HSAKMT_MODEL_ORDINARY_BACKING_BYTES +
        SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BASE_BYTES +
        (uint64_t)slot *
            SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_SLOT_BYTES;
    local->last_doorbell_notification = 0;
    local->last_completion = 0;
    local->slot = slot;
    local->queue_percentage = ioctl_args->queue_percentage;
    local->queue_priority = ioctl_args->queue_priority;
    local->handle = queue_handle;
    if (model_start_queue_progress_locked() != 0) {
      const int saved_errno = errno != 0 ? errno : EIO;
      (void)sagr_kmt_queue_destroy(model_provider, &model_kfd,
                                   &local->handle, NULL, NULL, 0U);
      memset(local, 0, sizeof(*local));
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    ioctl_args->queue_id = local->raw_queue_id;
    ioctl_args->doorbell_offset = local->doorbell_offset;
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_UPDATE_QUEUE) {
    const struct kfd_ioctl_update_queue_args *ioctl_args = argument;
    struct model_queue *local;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    if (ioctl_args->ring_base_address == 0U ||
        (ioctl_args->ring_base_address & UINT64_C(63)) != 0U ||
        ioctl_args->ring_size < UINT32_C(4096) ||
        (ioctl_args->ring_size & (ioctl_args->ring_size - 1U)) != 0U ||
        (ioctl_args->ring_size % SAGR_HSAKMT_MODEL_AQL_PACKET_BYTES) != 0U ||
        ((uint64_t)ioctl_args->ring_size /
             SAGR_HSAKMT_MODEL_AQL_PACKET_BYTES) >
            (uint64_t)SAGR_KMT_QUEUE_MAX_DEPTH ||
        ioctl_args->queue_percentage > KFD_MAX_QUEUE_PERCENTAGE ||
        ioctl_args->queue_priority > KFD_MAX_QUEUE_PRIORITY) {
      errno = EINVAL;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    local = model_queue_locked(ioctl_args->queue_id);
    if (local == NULL || model_owner_pid != getpid()) {
      (void)pthread_mutex_unlock(&model_mutex);
      errno = model_owner_pid != getpid() ? EOWNERDEAD : EBADF;
      return -1;
    }
    /* Queue configuration is consumed by the future AQL executor.  Do not
     * emit a made-up transport operation for an ioctl that the bridge wire
     * does not carry; commit only the authenticated local state. */
    local->ring_base_address = ioctl_args->ring_base_address;
    local->ring_size_bytes = ioctl_args->ring_size;
    local->queue_percentage = ioctl_args->queue_percentage;
    local->queue_priority = ioctl_args->queue_priority;
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_DESTROY_QUEUE) {
    const struct kfd_ioctl_destroy_queue_args *ioctl_args = argument;
    struct model_queue *local;
    sagr_kmt_status_t status;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0) {
      errno = model_atfork_error != 0 ? model_atfork_error : EIO;
      return -1;
    }
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) {
      errno = lock_status;
      return -1;
    }
    local = model_queue_locked(ioctl_args->queue_id);
    if (local == NULL || model_owner_pid != getpid() ||
        model_connect_locked() != 0) {
      const int saved_errno = model_owner_pid != getpid()
                                  ? EOWNERDEAD
                                  : (errno != 0 ? errno : EBADF);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    status = sagr_kmt_queue_destroy(model_provider, &model_kfd,
                                    &local->handle, NULL, NULL, 0U);
    if (status != SAGR_KMT_STATUS_SUCCESS) {
      const int saved_errno = model_kmt_status_errno(status);
      (void)pthread_mutex_unlock(&model_mutex);
      errno = saved_errno;
      return -1;
    }
    memset(local, 0, sizeof(*local));
    (void)pthread_mutex_unlock(&model_mutex);
    return 0;
  }
  if (request == AMDKFD_IOC_SET_SCRATCH_BACKING_VA) {
    const struct kfd_ioctl_set_scratch_backing_va_args *ioctl_args = argument;
    uint64_t scratch_backing_va;
    sagr_kmt_status_t status;
    if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
        model_atfork_error != 0 ||
        ioctl_args->gpu_id != SAGR_HSAKMT_MODEL_VISIBLE_GPU_ID ||
        ioctl_args->va_addr == 0U ||
        ioctl_args->va_addr > (UINT64_MAX >> 16U)) {
      errno = EINVAL;
      return -1;
    }
    /* KFD carries SH_HIDDEN_PRIVATE_BASE in 64 KiB units.  The typed
     * self-runtime contract carries the corresponding byte address. */
    scratch_backing_va = ioctl_args->va_addr << 16U;
    lock_status = pthread_mutex_lock(&model_mutex);
    if (lock_status != 0) { errno = lock_status; return -1; }
    if (model_owner_pid != getpid() || model_connect_locked() != 0) {
      const int saved_errno = model_owner_pid != getpid() ? EOWNERDEAD : (errno != 0 ? errno : EIO);
      (void)pthread_mutex_unlock(&model_mutex); errno = saved_errno; return -1;
    }
    status = sagr_kmt_set_scratch_backing_va(model_provider, &model_kfd,
                                              ioctl_args->gpu_id,
                                              scratch_backing_va,
                                              NULL, NULL, 0U);
    (void)pthread_mutex_unlock(&model_mutex);
    if (status != SAGR_KMT_STATUS_SUCCESS) { errno = model_kmt_status_errno(status); return -1; }
    return 0;
  }
  errno = ENOTSUP;
  return -1;
}

static int model_trace_enabled(void) {
  const char *value = getenv("SAGR_HSAKMT_MODEL_TRACE");
  return value != NULL && strcmp(value, "1") == 0;
}

static void model_trace_ioctl(const char *phase, unsigned long request,
                              const void *argument, int result,
                              int error_number) {
  if (model_trace_enabled() == 0) {
    return;
  }
  flockfile(stderr);
  fprintf(stderr,
          "hsakmt-model pid=%ld phase=%s request=0x%lx result=%d errno=%d",
          (long)getpid(), phase, request, result, error_number);
  if (argument != NULL && request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
    const struct kfd_ioctl_alloc_memory_of_gpu_args *args = argument;
    fprintf(stderr,
            " gpu_id=%u flags=0x%llx va=0x%llx size=%llu mmap_offset=0x%llx handle=%llu",
            args->gpu_id, (unsigned long long)args->flags,
            (unsigned long long)args->va_addr,
            (unsigned long long)args->size,
            (unsigned long long)args->mmap_offset,
            (unsigned long long)args->handle);
  } else if (argument != NULL && request == AMDKFD_IOC_CREATE_QUEUE) {
    const struct kfd_ioctl_create_queue_args *args = argument;
    fprintf(stderr,
            " gpu_id=%u queue_type=%u ring=0x%llx ring_size=%u read=0x%llx write=0x%llx percentage=%u priority=%u queue_id=%u doorbell_offset=0x%llx",
            args->gpu_id, args->queue_type,
            (unsigned long long)args->ring_base_address, args->ring_size,
            (unsigned long long)args->read_pointer_address,
            (unsigned long long)args->write_pointer_address,
            args->queue_percentage, args->queue_priority, args->queue_id,
            (unsigned long long)args->doorbell_offset);
  } else if (argument != NULL && request == AMDKFD_IOC_UPDATE_QUEUE) {
    const struct kfd_ioctl_update_queue_args *args = argument;
    fprintf(stderr,
            " queue_id=%u ring=0x%llx ring_size=%u percentage=%u priority=%u",
            args->queue_id, (unsigned long long)args->ring_base_address,
            args->ring_size, args->queue_percentage, args->queue_priority);
  } else if (argument != NULL && request == AMDKFD_IOC_DESTROY_QUEUE) {
    const struct kfd_ioctl_destroy_queue_args *args = argument;
    fprintf(stderr, " queue_id=%u", args->queue_id);
  }
  fputc('\n', stderr);
  funlockfile(stderr);
}

static int model_handle_ioctl(unsigned long request, void *argument) {
  int result;
  int saved_errno;
  model_trace_ioctl("enter", request, argument, 0, 0);
  result = model_handle_ioctl_impl(request, argument);
  saved_errno = errno;
  model_trace_ioctl("leave", request, argument, result,
                    result == 0 ? 0 : saved_errno);
  errno = saved_errno;
  return result;
}

static int model_handle_drm_call(unsigned command, void *argument) {
  int lock_status;
  int result = -1;
  int saved_errno = ENOTSUP;
  if (command > (unsigned)HSAKMT_DRM_QUERY_GPU_INFO || argument == NULL) {
    errno = EINVAL;
    return -1;
  }
  if (pthread_once(&model_atfork_once, model_install_atfork) != 0 ||
      model_atfork_error != 0) {
    errno = model_atfork_error != 0 ? model_atfork_error : EIO;
    return -1;
  }
  lock_status = pthread_mutex_lock(&model_mutex);
  if (lock_status != 0) {
    errno = lock_status;
    return -1;
  }
  if (model_owner_pid != getpid()) {
    saved_errno = EOWNERDEAD;
    goto out;
  }

  switch ((enum hsakmt_drm_cmd)command) {
    case HSAKMT_DRM_OPEN_RENDER: {
      struct hsakmt_drm_open_render_args *request = argument;
      int committed;
      struct model_render *render;
      if (request->fd_out == NULL ||
          request->minor < SAGR_HSAKMT_MODEL_RENDER_FIRST ||
          request->minor > SAGR_HSAKMT_MODEL_RENDER_LAST) {
        saved_errno = EINVAL;
        break;
      }
      if (model_memfd < 0) {
        model_memfd = model_create_storage();
        if (model_memfd < 0) {
          saved_errno = errno != 0 ? errno : EIO;
          break;
        }
      }
      committed = fcntl(model_memfd, F_DUPFD_CLOEXEC, 3);
      if (committed < 0) {
        saved_errno = errno != 0 ? errno : EMFILE;
        break;
      }
      render = model_allocate_render_locked(committed);
      if (render == NULL) {
        (void)close(committed);
        saved_errno = EMFILE;
        break;
      }
      render->active = 1;
      render->owner_pid = getpid();
      render->descriptor = committed;
      render->render_minor = request->minor;
      *request->fd_out = committed;
      result = 0;
      break;
    }
    case HSAKMT_DRM_CLOSE: {
      const struct hsakmt_drm_close_args *request = argument;
      struct model_render *render;
      if (request->fd == model_memfd ||
          !model_descriptor_is_storage_locked(request->fd)) {
        saved_errno = EBADF;
        break;
      }
      render = model_render_locked(request->fd);
      if (close(request->fd) != 0) {
        saved_errno = errno != 0 ? errno : EIO;
        break;
      }
      if (render != NULL) {
        memset(render, 0, sizeof(*render));
      }
      result = 0;
      break;
    }
    case HSAKMT_DRM_DEVICE_INITIALIZE: {
      struct hsakmt_drm_device_initialize_args *request = argument;
      struct model_device *device;
      struct model_render *render;
      int committed_descriptor;
      if (request->major_out == NULL || request->minor_out == NULL ||
          request->dev_out == NULL) {
        saved_errno = EINVAL;
        break;
      }
      render = model_render_locked(request->fd);
      if (render == NULL) {
        saved_errno = EBADF;
        break;
      }
      device = model_allocate_device_locked();
      if (device == NULL) {
        saved_errno = EMFILE;
        break;
      }
      if (model_next_device_token == (uintptr_t)0U) {
        saved_errno = EOVERFLOW;
        break;
      }
      committed_descriptor = fcntl(request->fd, F_DUPFD_CLOEXEC, 3);
      if (committed_descriptor < 0) {
        saved_errno = errno != 0 ? errno : EMFILE;
        break;
      }
      device->active = 1;
      device->owner_pid = getpid();
      device->descriptor = committed_descriptor;
      device->render_minor = render->render_minor;
      device->token = model_next_device_token++;
      *request->major_out = SAGR_HSAKMT_MODEL_DRM_MAJOR;
      *request->minor_out = SAGR_HSAKMT_MODEL_DRM_MINOR;
      *request->dev_out = (void *)device->token;
      result = 0;
      break;
    }
    case HSAKMT_DRM_DEVICE_DEINITIALIZE: {
      const struct hsakmt_drm_device_deinitialize_args *request = argument;
      struct model_device *device = model_device_locked(request->dev);
      if (device == NULL) {
        saved_errno = EBADF;
        break;
      }
      if (close(device->descriptor) != 0) {
        saved_errno = errno != 0 ? errno : EIO;
        break;
      }
      device->descriptor = -1;
      device->owner_pid = 0;
      device->render_minor = 0;
      device->token = (uintptr_t)0U;
      device->active = 0;
      result = 0;
      break;
    }
    case HSAKMT_DRM_DEVICE_GET_FD: {
      struct hsakmt_drm_device_get_fd_args *request = argument;
      struct model_device *device;
      int committed;
      struct model_render *render;
      if (request->fd_out == NULL) {
        saved_errno = EINVAL;
        break;
      }
      device = model_device_locked(request->dev);
      if (device == NULL) {
        saved_errno = EBADF;
        break;
      }
      committed = fcntl(device->descriptor, F_DUPFD_CLOEXEC, 3);
      if (committed < 0) {
        saved_errno = errno != 0 ? errno : EMFILE;
        break;
      }
      render = model_allocate_render_locked(committed);
      if (render == NULL) {
        (void)close(committed);
        saved_errno = EMFILE;
        break;
      }
      render->active = 1;
      render->owner_pid = getpid();
      render->descriptor = committed;
      render->render_minor = device->render_minor;
      *request->fd_out = committed;
      result = 0;
      break;
    }
    case HSAKMT_DRM_GET_MARKETING_NAME: {
      struct hsakmt_drm_get_marketing_name_args *request = argument;
      if (request->name_out == NULL || model_device_locked(request->dev) == NULL) {
        saved_errno = request->name_out == NULL ? EINVAL : EBADF;
        break;
      }
      *request->name_out = model_marketing_name;
      result = 0;
      break;
    }
    case HSAKMT_DRM_QUERY_GPU_INFO: {
      struct hsakmt_drm_query_gpu_info_args *request = argument;
      struct amdgpu_gpu_info committed;
      if (request->info_out == NULL || model_device_locked(request->dev) == NULL) {
        saved_errno = request->info_out == NULL ? EINVAL : EBADF;
        break;
      }
      memset(&committed, 0, sizeof(committed));
      committed.asic_id = SAGR_HSAKMT_MODEL_GFX950_DEVICE_ID;
      committed.family_id = SAGR_HSAKMT_MODEL_GFX950_FAMILY_ID;
      committed.max_engine_clk = UINT64_C(2400000);
      committed.max_memory_clk = UINT64_C(1600000);
      committed.num_shader_engines = 4U;
      committed.num_shader_arrays_per_engine = 2U;
      committed.gpu_counter_freq = 100000U;
      committed.cu_active_number = 256U;
      committed.vram_bit_width = 8192U;
      memcpy(request->info_out, &committed, sizeof(committed));
      result = 0;
      break;
    }
    case HSAKMT_DRM_BO_VA_OP:
    case HSAKMT_DRM_BO_FREE:
    case HSAKMT_DRM_BO_IMPORT:
    case HSAKMT_DRM_BO_EXPORT:
    case HSAKMT_DRM_BO_CPU_MAP:
    case HSAKMT_DRM_BO_QUERY_INFO:
    case HSAKMT_DRM_BO_SET_METADATA:
    case HSAKMT_DRM_COMMAND_WRITE_READ:
      saved_errno = ENOTSUP;
      break;
  }

out:
  (void)pthread_mutex_unlock(&model_mutex);
  if (result != 0) {
    errno = saved_errno;
  }
  return result;
}

static const struct hsakmt_model_functions model_functions = {
    HSAKMT_MODEL_INTERFACE_VERSION_MAJOR,
    HSAKMT_MODEL_INTERFACE_VERSION_MINOR,
    model_create_memfd,
    model_handle_ioctl,
    model_handle_drm_call,
};

__attribute__((visibility("default")))
const struct hsakmt_model_functions *get_hsakmt_model_functions(void) {
  return &model_functions;
}

__attribute__((destructor)) static void model_destroy(void) {
  pthread_t progress_thread;
  int join_progress = 0;
  if (pthread_mutex_lock(&model_mutex) != 0) {
    return;
  }
  if (model_owner_pid == getpid() && model_progress_started != 0) {
    model_progress_stop = 1;
    progress_thread = model_progress_thread;
    join_progress = 1;
  }
  (void)pthread_mutex_unlock(&model_mutex);
  if (join_progress != 0) {
    (void)pthread_join(progress_thread, NULL);
  }
  if (pthread_mutex_lock(&model_mutex) != 0) {
    return;
  }
  if (model_owner_pid == getpid()) {
    model_progress_started = 0;
    model_disconnect_locked();
    model_reset_devices_locked();
    if (model_memfd >= 0) {
      (void)close(model_memfd);
      model_memfd = -1;
    }
  }
  (void)pthread_mutex_unlock(&model_mutex);
}
