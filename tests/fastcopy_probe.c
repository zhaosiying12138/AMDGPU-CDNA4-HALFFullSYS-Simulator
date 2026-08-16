/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#define FASTCOPY_PROBE_BYTES ((size_t)2U * 1024U * 1024U)
#define FASTCOPY_COMPLETION_TIMEOUT UINT64_C(5000000000)

typedef struct agent_search {
  hsa_agent_t cpu;
  hsa_agent_t gpu;
  hsa_status_t first_error;
} agent_search_t;

typedef struct pool_search {
  hsa_amd_memory_pool_t selected;
  hsa_amd_memory_pool_location_t location;
  uint32_t selected_flags;
  bool prefer_coarse;
  bool selected_is_preferred;
  hsa_status_t first_error;
} pool_search_t;

typedef struct probe_resources {
  void *host_source;
  void *host_destination;
  void *gpu_buffer;
  hsa_signal_t dependency;
  hsa_signal_t completion;
  bool async_submitted;
  bool dependency_created;
  bool completion_created;
  bool runtime_initialized;
} probe_resources_t;

static hsa_status_t visit_agent(hsa_agent_t agent, void *opaque) {
  agent_search_t *search = (agent_search_t *)opaque;
  hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status;

  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) {
    if (search->first_error == HSA_STATUS_SUCCESS) {
      search->first_error = status;
    }
    return status;
  }
  if (type == HSA_DEVICE_TYPE_CPU && search->cpu.handle == 0U) {
    search->cpu = agent;
  } else if (type == HSA_DEVICE_TYPE_GPU && search->gpu.handle == 0U) {
    search->gpu = agent;
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t visit_pool(hsa_amd_memory_pool_t pool, void *opaque) {
  pool_search_t *search = (pool_search_t *)opaque;
  hsa_amd_segment_t segment = HSA_AMD_SEGMENT_GLOBAL;
  hsa_amd_memory_pool_location_t location = HSA_AMD_MEMORY_POOL_LOCATION_CPU;
  uint32_t flags = 0U;
  bool allocation_allowed = false;
  bool preferred;
  hsa_status_t status;

  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status == HSA_STATUS_SUCCESS) {
    status = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
        &allocation_allowed);
  }
  if (status == HSA_STATUS_SUCCESS &&
      segment == HSA_AMD_SEGMENT_GLOBAL && allocation_allowed) {
    status = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_LOCATION, &location);
  }
  if (status == HSA_STATUS_SUCCESS &&
      segment == HSA_AMD_SEGMENT_GLOBAL && allocation_allowed) {
    status = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  }
  if (status != HSA_STATUS_SUCCESS) {
    if (search->first_error == HSA_STATUS_SUCCESS) {
      search->first_error = status;
    }
    return status;
  }
  if (segment != HSA_AMD_SEGMENT_GLOBAL || !allocation_allowed ||
      location != search->location) {
    return HSA_STATUS_SUCCESS;
  }

  preferred = search->prefer_coarse
                  ? (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) !=
                        0U
                  : (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) !=
                        0U;
  if (search->selected.handle == 0U ||
      (preferred && !search->selected_is_preferred)) {
    search->selected = pool;
    search->selected_flags = flags;
    search->selected_is_preferred = preferred;
  }
  return HSA_STATUS_SUCCESS;
}

static int find_pool(hsa_agent_t agent,
                     hsa_amd_memory_pool_location_t location,
                     bool prefer_coarse, pool_search_t *search) {
  hsa_status_t status;

  memset(search, 0, sizeof(*search));
  search->location = location;
  search->prefer_coarse = prefer_coarse;
  search->first_error = HSA_STATUS_SUCCESS;
  status = hsa_amd_agent_iterate_memory_pools(agent, visit_pool, search);
  return status == HSA_STATUS_SUCCESS &&
                 search->first_error == HSA_STATUS_SUCCESS &&
                 search->selected.handle != 0U
             ? 0
             : -1;
}

static uint8_t pattern_byte(size_t index, uint8_t salt) {
  return (uint8_t)((uint8_t)index ^ (uint8_t)(index >> 8U) ^ salt);
}

static void fill_pattern(uint8_t *buffer, size_t size, uint8_t salt) {
  size_t index;
  for (index = 0U; index < size; ++index) {
    buffer[index] = pattern_byte(index, salt);
  }
}

static int expect_status(hsa_status_t status, const char *operation) {
  if (status == HSA_STATUS_SUCCESS) {
    return 0;
  }
  fprintf(stderr, "fastcopy probe: %s failed: status=%u\n", operation,
          (unsigned)status);
  return -1;
}

static void release_resources(probe_resources_t *resources) {
  if (resources->async_submitted && resources->dependency_created &&
      resources->completion_created) {
    /* Keep cleanup ordered if a negative-path test exits while the copy is
     * still waiting on its dependency. */
    hsa_signal_store_screlease(resources->dependency, 0);
    (void)hsa_signal_wait_scacquire(
        resources->completion, HSA_SIGNAL_CONDITION_LT, 1,
        FASTCOPY_COMPLETION_TIMEOUT, HSA_WAIT_STATE_BLOCKED);
  }
  if (resources->completion_created) {
    (void)hsa_signal_destroy(resources->completion);
  }
  if (resources->dependency_created) {
    (void)hsa_signal_destroy(resources->dependency);
  }
  if (resources->gpu_buffer != NULL) {
    (void)hsa_amd_memory_pool_free(resources->gpu_buffer);
  }
  if (resources->host_destination != NULL) {
    (void)hsa_amd_memory_pool_free(resources->host_destination);
  }
  if (resources->host_source != NULL) {
    (void)hsa_amd_memory_pool_free(resources->host_source);
  }
  if (resources->runtime_initialized) {
    (void)hsa_shut_down();
  }
}

static int dependency_probe_enabled(int argc, char **argv) {
  if (argc == 1) {
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "--dependency") == 0) {
    return 1;
  }
  fprintf(stderr, "usage: %s [--dependency]\n", argv[0]);
  return -1;
}

int main(int argc, char **argv) {
  agent_search_t agents;
  pool_search_t cpu_pool;
  pool_search_t gpu_pool;
  probe_resources_t resources;
  hsa_agent_t allowed_agents[2];
  hsa_signal_value_t completion_value;
  const char *runtime_gate = getenv("HSA_ENABLE_DTIF_FAST_COPY");
  const char *provider_gate = getenv("SAGR_HSAKMT_MODEL_FAST_COPY");
  const char *async_result = "skipped";
  hsa_status_t status;
  const int run_dependency = dependency_probe_enabled(argc, argv);
  int result = EXIT_FAILURE;

  if (run_dependency < 0) {
    return EXIT_FAILURE;
  }

  memset(&agents, 0, sizeof(agents));
  memset(&resources, 0, sizeof(resources));
  agents.first_error = HSA_STATUS_SUCCESS;

  status = hsa_init();
  if (expect_status(status, "hsa_init") != 0) {
    goto out;
  }
  resources.runtime_initialized = true;
  status = hsa_iterate_agents(visit_agent, &agents);
  if (expect_status(status, "hsa_iterate_agents") != 0 ||
      agents.first_error != HSA_STATUS_SUCCESS || agents.cpu.handle == 0U ||
      agents.gpu.handle == 0U) {
    fprintf(stderr, "fastcopy probe: one CPU and one GPU agent are required\n");
    goto out;
  }
  if (find_pool(agents.cpu, HSA_AMD_MEMORY_POOL_LOCATION_CPU, false,
                &cpu_pool) != 0 ||
      find_pool(agents.gpu, HSA_AMD_MEMORY_POOL_LOCATION_GPU, true, &gpu_pool) !=
          0) {
    fprintf(stderr,
            "fastcopy probe: allocatable CPU and GPU global pools are required\n");
    goto out;
  }

  status = hsa_amd_memory_pool_allocate(
      cpu_pool.selected, FASTCOPY_PROBE_BYTES, 0U, &resources.host_source);
  if (expect_status(status, "CPU source allocation") != 0 ||
      resources.host_source == NULL) {
    goto out;
  }
  status = hsa_amd_memory_pool_allocate(
      cpu_pool.selected, FASTCOPY_PROBE_BYTES, 0U,
      &resources.host_destination);
  if (expect_status(status, "CPU destination allocation") != 0 ||
      resources.host_destination == NULL) {
    goto out;
  }
  status = hsa_amd_memory_pool_allocate(
      gpu_pool.selected, FASTCOPY_PROBE_BYTES, 0U, &resources.gpu_buffer);
  if (expect_status(status, "GPU public allocation") != 0 ||
      resources.gpu_buffer == NULL) {
    goto out;
  }

  allowed_agents[0] = agents.cpu;
  allowed_agents[1] = agents.gpu;
  status = hsa_amd_agents_allow_access(
      2U, allowed_agents, NULL, resources.host_source);
  if (status == HSA_STATUS_SUCCESS) {
    status = hsa_amd_agents_allow_access(
        2U, allowed_agents, NULL, resources.host_destination);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = hsa_amd_agents_allow_access(
        2U, allowed_agents, NULL, resources.gpu_buffer);
  }
  if (expect_status(status, "allocation access publication") != 0) {
    goto out;
  }

  fill_pattern((uint8_t *)resources.host_source, FASTCOPY_PROBE_BYTES, 0x5aU);
  memset(resources.host_destination, 0, FASTCOPY_PROBE_BYTES);
  status = hsa_memory_copy(resources.gpu_buffer, resources.host_source,
                           FASTCOPY_PROBE_BYTES);
  if (expect_status(status, "synchronous H2D copy") != 0) {
    goto out;
  }
  status = hsa_memory_copy(resources.host_destination, resources.gpu_buffer,
                           FASTCOPY_PROBE_BYTES);
  if (expect_status(status, "synchronous D2H copy") != 0) {
    goto out;
  }
  if (memcmp(resources.host_source, resources.host_destination,
             FASTCOPY_PROBE_BYTES) != 0) {
    fprintf(stderr, "fastcopy probe: synchronous roundtrip mismatch\n");
    goto out;
  }

  if (run_dependency != 0) {
    /* A ready dependency exercises the dependency-bearing API without
     * depending on the model bridge's blocked-worker wakeup behavior. */
    status = hsa_signal_create(0, 0U, NULL, &resources.dependency);
    if (expect_status(status, "dependency signal creation") != 0) {
      goto out;
    }
    resources.dependency_created = true;
    status = hsa_signal_create(1, 0U, NULL, &resources.completion);
    if (expect_status(status, "completion signal creation") != 0) {
      goto out;
    }
    resources.completion_created = true;

    fill_pattern((uint8_t *)resources.host_source, FASTCOPY_PROBE_BYTES, 0xa5U);
    status = hsa_amd_memory_async_copy(
        resources.gpu_buffer, agents.gpu, resources.host_source, agents.cpu,
        FASTCOPY_PROBE_BYTES, 1U, &resources.dependency,
        resources.completion);
    if (status != HSA_STATUS_SUCCESS) {
      fprintf(stderr,
              "fastcopy probe: async dependency path skipped: status=%u\n",
              (unsigned)status);
      resources.completion_created = false;
      (void)hsa_signal_destroy(resources.completion);
      resources.dependency_created = false;
      (void)hsa_signal_destroy(resources.dependency);
    } else {
      resources.async_submitted = true;
      completion_value = hsa_signal_wait_scacquire(
          resources.completion, HSA_SIGNAL_CONDITION_LT, 1,
          FASTCOPY_COMPLETION_TIMEOUT, HSA_WAIT_STATE_BLOCKED);
      if (completion_value != 0) {
        fprintf(stderr,
                "fastcopy probe: async dependency path skipped: "
                "completion=%lld\n",
                (long long)completion_value);
      } else {
        memset(resources.host_destination, 0, FASTCOPY_PROBE_BYTES);
        status = hsa_memory_copy(resources.host_destination,
                                 resources.gpu_buffer, FASTCOPY_PROBE_BYTES);
        if (expect_status(status, "post-async synchronous D2H copy") != 0 ||
            memcmp(resources.host_source, resources.host_destination,
                   FASTCOPY_PROBE_BYTES) != 0) {
          fprintf(stderr,
                  "fastcopy probe: dependency-bearing async copy mismatch\n");
          goto out;
        }
        async_result = "exact_ready_dependency";
      }
    }
  }

  printf("fastcopy probe passed: bytes=%zu sync_h2d=exact sync_d2h=exact "
         "async_dependency=%s runtime_gate=%s provider_gate=%s "
         "cpu_pool_flags=0x%x gpu_pool_flags=0x%x\n",
         FASTCOPY_PROBE_BYTES,
         async_result,
         runtime_gate != NULL ? runtime_gate : "unset",
         provider_gate != NULL ? provider_gate : "unset",
         cpu_pool.selected_flags, gpu_pool.selected_flags);
  result = EXIT_SUCCESS;

out:
  release_resources(&resources);
  return result;
}
