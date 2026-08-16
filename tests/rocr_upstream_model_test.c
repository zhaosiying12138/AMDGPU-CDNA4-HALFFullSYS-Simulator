/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#endif
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#include <hsakmt/hsakmtmodeliface.h>

#define EXECUTION_ELEMENT_COUNT 64U
#define EXECUTION_QUEUE_SIZE 64U
#define EXECUTION_TIMEOUT UINT64_C(5000000000)
#define MAXIMUM_CODE_OBJECT_BYTES UINT64_C(67108864)
#define MODEL_LOCAL_MEMORY_BYTES UINT64_C(309237645312)

typedef struct agent_summary {
  unsigned total;
  unsigned cpu;
  unsigned gpu;
  unsigned other;
  hsa_status_t first_error;
  hsa_agent_t first_cpu;
  hsa_agent_t first_gpu;
} agent_summary_t;

typedef struct pool_search {
  hsa_amd_memory_pool_t standard;
  hsa_amd_memory_pool_t kernarg;
  hsa_status_t first_error;
} pool_search_t;

typedef struct execution_resources {
  int code_object_fd;
  hsa_code_object_reader_t reader;
  hsa_executable_t executable;
  hsa_signal_t signal;
  hsa_queue_t *queue;
  int *input;
  int *system_output;
  int *gpu_output;
  void *kernarg;
  int reader_created;
  int executable_created;
  int signal_created;
} execution_resources_t;

static void execution_phase(const char *phase) {
  const char *enabled = getenv("SAGR_UPSTREAM_ROCR_EXECUTION_TRACE");
  if (enabled != NULL && strcmp(enabled, "1") == 0) {
    fprintf(stderr, "upstream-rocr-execution phase=%s\n", phase);
  }
}

static hsa_status_t visit_agent(hsa_agent_t agent, void *opaque) {
  agent_summary_t *summary = (agent_summary_t *)opaque;
  hsa_device_type_t device = HSA_DEVICE_TYPE_CPU;
  char name[64];
  hsa_status_t status;

  memset(name, 0, sizeof(name));
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device);
  if (status != HSA_STATUS_SUCCESS) {
    if (summary->first_error == HSA_STATUS_SUCCESS) {
      summary->first_error = status;
    }
    return status;
  }
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  if (status != HSA_STATUS_SUCCESS) {
    if (summary->first_error == HSA_STATUS_SUCCESS) {
      summary->first_error = status;
    }
    return status;
  }
  ++summary->total;
  switch (device) {
    case HSA_DEVICE_TYPE_CPU:
      ++summary->cpu;
      if (summary->first_cpu.handle == 0U) {
        summary->first_cpu = agent;
      }
      break;
    case HSA_DEVICE_TYPE_GPU:
      ++summary->gpu;
      if (summary->first_gpu.handle == 0U) {
        summary->first_gpu = agent;
      }
      break;
    default:
      ++summary->other;
      break;
  }
  printf("agent device=%u name=%s\n", (unsigned)device, name);
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t visit_pool(hsa_amd_memory_pool_t pool, void *opaque) {
  pool_search_t *search = (pool_search_t *)opaque;
  hsa_amd_segment_t segment = HSA_AMD_SEGMENT_GLOBAL;
  uint32_t flags = 0U;
  bool alloc_allowed = false;
  hsa_status_t status;

  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    if (search->first_error == HSA_STATUS_SUCCESS) {
      search->first_error = status;
    }
    return status;
  }
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
  if (status != HSA_STATUS_SUCCESS) {
    if (search->first_error == HSA_STATUS_SUCCESS) {
      search->first_error = status;
    }
    return status;
  }
  if (segment != HSA_AMD_SEGMENT_GLOBAL || !alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (status != HSA_STATUS_SUCCESS) {
    if (search->first_error == HSA_STATUS_SUCCESS) {
      search->first_error = status;
    }
    return status;
  }
  if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0U) {
    if (search->kernarg.handle == 0U) {
      search->kernarg = pool;
    }
  } else if (search->standard.handle == 0U) {
    search->standard = pool;
  }
  return HSA_STATUS_SUCCESS;
}

static int find_pools(hsa_agent_t agent, pool_search_t *search) {
  hsa_status_t status;
  memset(search, 0, sizeof(*search));
  search->first_error = HSA_STATUS_SUCCESS;
  status = hsa_amd_agent_iterate_memory_pools(agent, visit_pool, search);
  return status == HSA_STATUS_SUCCESS &&
         search->first_error == HSA_STATUS_SUCCESS
             ? 0
             : -1;
}

static int wait_for_bridge_retirement(hsa_queue_t *queue,
                                      uint64_t completion) {
  uint64_t observed = 0;
  unsigned attempt;

  for (attempt = 0; attempt < 2000U; ++attempt) {
    const struct timespec delay = {0, 1000000};
    observed = hsa_queue_load_read_index_scacquire(queue);
    if (observed == completion) {
      break;
    }
    (void)nanosleep(&delay, NULL);
  }
  return observed == completion ? 0 : -1;
}

static void release_execution_resources(execution_resources_t *resources) {
  if (resources->signal_created != 0) {
    (void)hsa_signal_destroy(resources->signal);
  }
  if (resources->queue != NULL) {
    (void)hsa_queue_destroy(resources->queue);
  }
  if (resources->kernarg != NULL) {
    (void)hsa_amd_memory_pool_free(resources->kernarg);
  }
  if (resources->input != NULL) {
    (void)hsa_amd_memory_pool_free(resources->input);
  }
  if (resources->system_output != NULL) {
    (void)hsa_amd_memory_pool_free(resources->system_output);
  }
  if (resources->gpu_output != NULL) {
    (void)hsa_amd_memory_pool_free(resources->gpu_output);
  }
  if (resources->executable_created != 0) {
    (void)hsa_executable_destroy(resources->executable);
  }
  if (resources->reader_created != 0) {
    (void)hsa_code_object_reader_destroy(resources->reader);
  }
  if (resources->code_object_fd >= 0) {
    (void)close(resources->code_object_fd);
  }
}

static int open_code_object(const char *path, int *descriptor) {
  struct stat metadata;
  int committed;

  if (path == NULL || path[0] != '/') {
    fprintf(stderr, "code object path must be absolute\n");
    return -1;
  }
  committed = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (committed < 0) {
    fprintf(stderr, "code object open failed: errno=%d\n", errno);
    return -1;
  }
  if (fstat(committed, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
      metadata.st_size <= 0 ||
      (uint64_t)metadata.st_size > MAXIMUM_CODE_OBJECT_BYTES) {
    fprintf(stderr, "code object must be a bounded regular file\n");
    (void)close(committed);
    return -1;
  }
  *descriptor = committed;
  return 0;
}

static int execute_gpu_read_write(const agent_summary_t *summary,
                                  const char *code_object_path) {
  execution_resources_t resources;
  pool_search_t cpu_pools;
  pool_search_t gpu_pools;
  hsa_executable_symbol_t symbol;
  hsa_agent_t allowed_agents[2];
  hsa_kernel_dispatch_packet_t *packet;
  hsa_status_t status;
  hsa_signal_value_t signal_value;
  uint64_t kernel_object = 0U;
  uint64_t write_index;
  uint32_t kernarg_size = 0U;
  uint32_t kernarg_alignment = 0U;
  uint32_t group_segment_size = 0U;
  uint32_t private_segment_size = 0U;
  int gpu_result[EXECUTION_ELEMENT_COUNT];
  const size_t payload_bytes = sizeof(gpu_result);
  unsigned index;
  int result = -1;

  memset(&resources, 0, sizeof(resources));
  resources.code_object_fd = -1;
  execution_phase("pool-discovery");
  if (summary->first_cpu.handle == 0U ||
      find_pools(summary->first_cpu, &cpu_pools) != 0 ||
      find_pools(summary->first_gpu, &gpu_pools) != 0 ||
      cpu_pools.kernarg.handle == 0U || gpu_pools.standard.handle == 0U) {
    fprintf(stderr, "required CPU kernarg or GPU global pool is absent\n");
    goto out;
  }
  if (open_code_object(code_object_path, &resources.code_object_fd) != 0) {
    goto out;
  }
  execution_phase("code-object-reader");
  status = hsa_code_object_reader_create_from_file(
      resources.code_object_fd, &resources.reader);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "code object reader creation failed: %u\n",
            (unsigned)status);
    goto out;
  }
  resources.reader_created = 1;
  status = hsa_executable_create_alt(
      HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
      &resources.executable);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "executable creation failed: %u\n", (unsigned)status);
    goto out;
  }
  resources.executable_created = 1;
  status = hsa_executable_load_agent_code_object(
      resources.executable, summary->first_gpu, resources.reader, NULL, NULL);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "code object load failed: %u\n", (unsigned)status);
    goto out;
  }
  execution_phase("executable-freeze");
  status = hsa_executable_freeze(resources.executable, NULL);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "executable freeze failed: %u\n", (unsigned)status);
    goto out;
  }
  execution_phase("symbol-query");
  status = hsa_executable_get_symbol_by_name(
      resources.executable, "gpuReadWrite.kd", &summary->first_gpu, &symbol);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "gpuReadWrite.kd lookup failed: %u\n", (unsigned)status);
    goto out;
  }
#define QUERY_KERNEL_INFO(attribute, destination)                              \
  do {                                                                         \
    status = hsa_executable_symbol_get_info(symbol, (attribute),              \
                                             (destination));                   \
    if (status != HSA_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "kernel metadata query failed: attribute=%u status=%u\n", \
              (unsigned)(attribute), (unsigned)status);                        \
      goto out;                                                                \
    }                                                                          \
  } while (0)
  QUERY_KERNEL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                    &kernel_object);
  QUERY_KERNEL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                    &kernarg_size);
  QUERY_KERNEL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                    &kernarg_alignment);
  QUERY_KERNEL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                    &group_segment_size);
  QUERY_KERNEL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                    &private_segment_size);
#undef QUERY_KERNEL_INFO
  if (kernel_object == 0U || kernarg_size < 24U || kernarg_size > 4096U ||
      kernarg_alignment == 0U ||
      (kernarg_alignment & (kernarg_alignment - 1U)) != 0U) {
    fprintf(stderr,
            "kernel metadata is outside the execution contract: object=%llu size=%u alignment=%u\n",
            (unsigned long long)kernel_object, kernarg_size,
            kernarg_alignment);
    goto out;
  }

  execution_phase("memory-allocate");
  status = hsa_amd_memory_pool_allocate(
      cpu_pools.kernarg, payload_bytes, 0U, (void **)&resources.input);
  if (status != HSA_STATUS_SUCCESS || resources.input == NULL) {
    fprintf(stderr, "input allocation failed: %u\n", (unsigned)status);
    goto out;
  }
  status = hsa_amd_memory_pool_allocate(
      cpu_pools.kernarg, payload_bytes, 0U,
      (void **)&resources.system_output);
  if (status != HSA_STATUS_SUCCESS || resources.system_output == NULL) {
    fprintf(stderr, "system output allocation failed: %u\n", (unsigned)status);
    goto out;
  }
  status = hsa_amd_memory_pool_allocate(
      gpu_pools.standard, payload_bytes, 0U, (void **)&resources.gpu_output);
  if (status != HSA_STATUS_SUCCESS || resources.gpu_output == NULL) {
    fprintf(stderr, "GPU output allocation failed: %u\n", (unsigned)status);
    goto out;
  }
  status = hsa_amd_memory_pool_allocate(
      cpu_pools.kernarg, (size_t)kernarg_size, 0U, &resources.kernarg);
  if (status != HSA_STATUS_SUCCESS || resources.kernarg == NULL) {
    fprintf(stderr, "kernarg allocation failed: %u\n", (unsigned)status);
    goto out;
  }
  allowed_agents[0] = summary->first_cpu;
  allowed_agents[1] = summary->first_gpu;
  status = hsa_amd_agents_allow_access(
      2U, allowed_agents, NULL, resources.input);
  if (status == HSA_STATUS_SUCCESS) {
    status = hsa_amd_agents_allow_access(
        2U, allowed_agents, NULL, resources.system_output);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = hsa_amd_agents_allow_access(
        2U, allowed_agents, NULL, resources.gpu_output);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = hsa_amd_agents_allow_access(
        2U, allowed_agents, NULL, resources.kernarg);
  }
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "allocation access publication failed: %u\n",
            (unsigned)status);
    goto out;
  }
  execution_phase("memory-initialize");
  for (index = 0U; index < EXECUTION_ELEMENT_COUNT; ++index) {
    resources.input[index] = (int)(index * 13U + 7U);
    resources.system_output[index] = -1;
    gpu_result[index] = -1;
  }
  memset(resources.kernarg, 0, (size_t)kernarg_size);
  memcpy((uint8_t *)resources.kernarg + 0U, &resources.input,
         sizeof(resources.input));
  memcpy((uint8_t *)resources.kernarg + 8U, &resources.system_output,
         sizeof(resources.system_output));
  memcpy((uint8_t *)resources.kernarg + 16U, &resources.gpu_output,
         sizeof(resources.gpu_output));
  memset(gpu_result, 0, sizeof(gpu_result));
  status = hsa_memory_copy(resources.gpu_output, gpu_result, payload_bytes);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "GPU output initialization failed: %u\n",
            (unsigned)status);
    goto out;
  }

  execution_phase("queue-create");
  status = hsa_queue_create(
      summary->first_gpu, EXECUTION_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, NULL,
      NULL, UINT32_MAX, UINT32_MAX, &resources.queue);
  if (status != HSA_STATUS_SUCCESS || resources.queue == NULL) {
    fprintf(stderr, "execution queue creation failed: %u\n",
            (unsigned)status);
    goto out;
  }
  status = hsa_signal_create(1, 0U, NULL, &resources.signal);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "completion signal creation failed: %u\n",
            (unsigned)status);
    goto out;
  }
  resources.signal_created = 1;

  execution_phase("packet-publish");
  write_index = hsa_queue_add_write_index_relaxed(resources.queue, 1U);
  packet = (hsa_kernel_dispatch_packet_t *)resources.queue->base_address +
           (write_index & (resources.queue->size - 1U));
  memset(packet, 0, sizeof(*packet));
  packet->setup = 1U << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet->workgroup_size_x = EXECUTION_ELEMENT_COUNT;
  packet->workgroup_size_y = 1U;
  packet->workgroup_size_z = 1U;
  packet->grid_size_x = EXECUTION_ELEMENT_COUNT;
  packet->grid_size_y = 1U;
  packet->grid_size_z = 1U;
  packet->private_segment_size = private_segment_size;
  packet->group_segment_size = group_segment_size;
  packet->kernel_object = kernel_object;
  packet->kernarg_address = resources.kernarg;
  packet->completion_signal = resources.signal;
  {
    uint16_t header =
        (uint16_t)((uint16_t)HSA_PACKET_TYPE_KERNEL_DISPATCH
                   << HSA_PACKET_HEADER_TYPE);
    header = (uint16_t)(header |
        ((uint16_t)1U << HSA_PACKET_HEADER_BARRIER));
    header = (uint16_t)(header |
        ((uint16_t)HSA_FENCE_SCOPE_SYSTEM
         << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE));
    header = (uint16_t)(header |
        ((uint16_t)HSA_FENCE_SCOPE_SYSTEM
         << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE));
    __atomic_store_n((uint16_t *)packet, header, __ATOMIC_RELEASE);
  }
  hsa_signal_store_screlease(resources.queue->doorbell_signal,
                             (hsa_signal_value_t)write_index);
  execution_phase("signal-wait");
  signal_value = hsa_signal_wait_scacquire(
      resources.signal, HSA_SIGNAL_CONDITION_LT, 1, EXECUTION_TIMEOUT,
      HSA_WAIT_STATE_BLOCKED);
  if (signal_value != 0 ||
      wait_for_bridge_retirement(resources.queue, write_index + 1U) != 0) {
    fprintf(stderr,
            "standard AQL dispatch did not retire: signal=%lld write=%llu\n",
            (long long)signal_value, (unsigned long long)write_index);
    goto out;
  }
  execution_phase("readback");
  status = hsa_memory_copy(gpu_result, resources.gpu_output, payload_bytes);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "GPU output readback failed: %u\n", (unsigned)status);
    goto out;
  }
  for (index = 0U; index < EXECUTION_ELEMENT_COUNT; ++index) {
    const int expected_input = (int)(index * 13U + 7U);
    if (resources.input[index] != expected_input ||
        resources.system_output[index] != (int)index ||
        gpu_result[index] != expected_input) {
      fprintf(stderr,
              "execution mismatch at %u: input=%d system=%d gpu=%d expected=%d\n",
              index, resources.input[index], resources.system_output[index],
              gpu_result[index], expected_input);
      goto out;
    }
  }
  execution_phase("verified");
  printf("upstream ROCr standard AQL execution passed: elements=%u kernel_object=0x%llx kernarg=%u\n",
         EXECUTION_ELEMENT_COUNT, (unsigned long long)kernel_object,
         kernarg_size);
  result = 0;

out:
  release_execution_resources(&resources);
  return result;
}

int main(int argc, char **argv) {
  agent_summary_t summary;
  hsa_queue_t *queue = NULL;
  hsa_status_t status;
  uint64_t available_memory = 0U;
  const char *execute_path = NULL;

  if (argc == 3 && strcmp(argv[1], "--execute") == 0) {
    execute_path = argv[2];
  } else if (argc != 1) {
    fprintf(stderr, "usage: %s [--execute /absolute/kernel.hsaco]\n", argv[0]);
    return 1;
  }

  memset(&summary, 0, sizeof(summary));
  summary.first_error = HSA_STATUS_SUCCESS;
  status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_init failed: %u\n", (unsigned)status);
    return 2;
  }
  status = hsa_iterate_agents(visit_agent, &summary);
  if (status != HSA_STATUS_SUCCESS || summary.first_error != HSA_STATUS_SUCCESS ||
      summary.total == 0U || summary.cpu == 0U || summary.gpu == 0U ||
      summary.first_cpu.handle == 0U || summary.first_gpu.handle == 0U) {
    fprintf(stderr,
            "agent enumeration failed: status=%u first_error=%u total=%u cpu=%u gpu=%u other=%u\n",
            (unsigned)status, (unsigned)summary.first_error, summary.total,
            summary.cpu, summary.gpu, summary.other);
    (void)hsa_shut_down();
    return 3;
  }
  status = hsa_agent_get_info(
      summary.first_gpu, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
      &available_memory);
  if (status != HSA_STATUS_SUCCESS ||
      available_memory != MODEL_LOCAL_MEMORY_BYTES) {
    fprintf(stderr,
            "available memory query failed: status=%u available=%llu\n",
            (unsigned)status, (unsigned long long)available_memory);
    (void)hsa_shut_down();
    return 4;
  }
  if (execute_path != NULL) {
    const int execution_status = execute_gpu_read_write(&summary, execute_path);
    status = hsa_shut_down();
    if (execution_status != 0 || status != HSA_STATUS_SUCCESS) {
      if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "hsa_shut_down failed: %u\n", (unsigned)status);
      }
      return 4;
    }
    return 0;
  }
  status = hsa_queue_create(summary.first_gpu, 64U, HSA_QUEUE_TYPE_MULTI,
                            NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
  if (status != HSA_STATUS_SUCCESS || queue == NULL || queue->size != 64U ||
      queue->base_address == NULL || queue->doorbell_signal.handle == 0U) {
    fprintf(stderr, "AQL queue creation failed: status=%u queue=%p\n",
            (unsigned)status, (void *)queue);
    (void)hsa_shut_down();
    return 4;
  }
  if (wait_for_bridge_retirement(queue, 1U) != 0) {
    fprintf(stderr, "queue completion shadow was not published upstream\n");
    (void)hsa_queue_destroy(queue);
    (void)hsa_shut_down();
    return 5;
  }
  status = hsa_queue_destroy(queue);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "AQL queue destroy failed: %u\n", (unsigned)status);
    (void)hsa_shut_down();
    return 6;
  }
  status = hsa_shut_down();
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_shut_down failed: %u\n", (unsigned)status);
    return 7;
  }
  printf("upstream ROCr model smoke passed: agents=%u gpu=%u queue=64\n",
         summary.total, summary.gpu);
  return 0;
}
