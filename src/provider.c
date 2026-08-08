/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/provider.h>

#include "provider_internal.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>

enum { SAGR_PROVIDER_MAGIC = 0x50525631 };

typedef struct provider_symbol_record {
  const char *name;
  uint32_t kind;
  uint32_t requirement;
  uint32_t layer;
  uint32_t version_script_exported;
} provider_symbol_record_t;

typedef struct provider_layout_record {
  const char *type_name;
  uint32_t size_bytes;
  uint32_t field_count;
} provider_layout_record_t;

typedef struct provider_layout_field_record {
  const char *name;
  uint32_t offset_bytes;
} provider_layout_field_record_t;

typedef struct provider_model_command_record {
  const char *name;
  const char *argument_type;
  uint32_t value;
  uint32_t size_bytes;
} provider_model_command_record_t;

static int valid_provider(const sagr_provider_t *provider);

sagr_instance_t sagr_provider_transport_instance(sagr_provider_t *provider) {
  return valid_provider(provider) ? provider->instance : NULL;
}

const sagr_instance_info_t *sagr_provider_transport_info(
    const sagr_provider_t *provider) {
  return valid_provider(provider) ? &provider->transport_info : NULL;
}

int sagr_provider_is_valid(const sagr_provider_t *provider) {
  return valid_provider(provider);
}

uint64_t sagr_provider_next_kmt_sequence(sagr_provider_t *provider) {
  uint64_t sequence;
  if (!valid_provider(provider)) {
    return 0;
  }
  sequence = atomic_fetch_add_explicit(&provider->kmt_operation_sequence,
                                       UINT64_C(1), memory_order_relaxed);
  if (sequence == 0) {
    sequence = atomic_fetch_add_explicit(&provider->kmt_operation_sequence,
                                         UINT64_C(1), memory_order_relaxed);
  }
  return sequence;
}

static const provider_symbol_record_t k_symbols[] = {
    {"hsaKmtOpenKFD", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"hsaKmtCloseKFD", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"hsaKmtGetVersion", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"hsaKmtAcquireSystemProperties", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"hsaKmtReleaseSystemProperties", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"hsaKmtGetNodeProperties", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"hsaKmtGetNodeMemoryProperties", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"hsaKmtGetNodeCacheProperties", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"hsaKmtGetNodeIoLinkProperties", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"hsaKmtCreateEvent", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_EVENT_SYNC, 1U},
    {"hsaKmtDestroyEvent", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_EVENT_SYNC, 1U},
    {"hsaKmtSetEvent", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_EVENT_SYNC, 1U},
    {"hsaKmtResetEvent", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_EVENT_SYNC, 1U},
    {"hsaKmtQueryEventState", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_EVENT_SYNC, 1U},
    {"hsaKmtWaitOnEvent", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_EVENT_SYNC, 1U},
    {"hsaKmtWaitOnMultipleEvents", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_EVENT_SYNC, 1U},
    {"hsaKmtCreateQueue", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 1U},
    {"hsaKmtCreateQueueExt", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 0U},
    {"hsaKmtCreateQueueV2", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 0U},
    {"hsaKmtUpdateQueue", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 1U},
    {"hsaKmtDestroyQueue", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 1U},
    {"hsaKmtSetQueueCUMask", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 1U},
    {"hsaKmtSetMemoryPolicy", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtAllocMemory", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtAllocMemoryAlign", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtFreeMemory", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtAvailableMemory", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtRegisterMemory", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtRegisterMemoryToNodes", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtRegisterMemoryWithFlags", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtRegisterGraphicsHandleToNodes", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtRegisterGraphicsHandleToNodesExt", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 0U},
    {"hsaKmtShareMemory", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtRegisterSharedHandle", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtRegisterSharedHandleToNodes", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtProcessVMRead", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtProcessVMWrite", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtDeregisterMemory", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtMapMemoryToGPU", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtMapMemoryToGPUNodes", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtUnmapMemoryToGPU", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtDbgRegister", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtDbgUnregister", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtDbgWavefrontControl", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtDbgAddressWatch", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtDbgEnable", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtDbgDisable", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtDbgGetDeviceData", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtDbgGetQueueData", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtGetClockCounters", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPmcGetCounterProperties", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPmcRegisterTrace", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPmcUnregisterTrace", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPmcAcquireTraceAccess", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPmcReleaseTraceAccess", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPmcStartTrace", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPmcQueryTrace", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPmcStopTrace", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtMapGraphicHandle", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtUnmapGraphicHandle", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtSetTrapHandler", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtSetSigbusDelay", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_OPTIONAL,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtGetTileConfig", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtQueryPointerInfo", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtSetMemoryUserData", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtGetQueueInfo", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 1U},
    {"hsaKmtGetKernelQueueId", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 1U},
    {"hsaKmtAllocQueueGWS", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 1U},
    {"hsaKmtRuntimeEnable", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtRuntimeDisable", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtCheckRuntimeDebugSupport", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtGetRuntimeCapabilities", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtGetCoreRuntimeInfo", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtGetCoreDeviceInfo", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtDebugTrapIoctl", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtSPMAcquire", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtSPMRelease", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtSPMSetDestBuffer", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtSVMSetAttr", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtSVMGetAttr", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtSetXNACKMode", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtGetXNACKMode", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtOpenSMI", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtExportDMABufHandle", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtWaitOnEvent_Ext", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_EVENT_SYNC, 1U},
    {"hsaKmtWaitOnMultipleEvents_Ext", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_EVENT_SYNC, 1U},
    {"hsaKmtReplaceAsanHeaderPage", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtReturnAsanHeaderPage", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtGetAMDGPUDeviceHandle", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtPcSamplingQueryCapabilities", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPcSamplingCreate", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPcSamplingDestroy", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPcSamplingStart", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPcSamplingStop", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtPcSamplingSupport", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtModelEnabled", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 0U},
    {"hsaKmtQueueRingDoorbell", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_QUEUE_DISPATCH, 0U},
    {"amdgpu_device_initialize", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"hsaKmtAisReadWriteFile", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY, 1U},
    {"hsaKmtGetMemoryHandle", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtHandleImport", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtImportExternalSemaphore", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_OPTIONAL, SAGR_PROVIDER_LAYER_EXTERNAL_SEMAPHORE, 1U},
    {"hsaKmtDestroyExternalSemaphore", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_OPTIONAL, SAGR_PROVIDER_LAYER_EXTERNAL_SEMAPHORE, 1U},
    {"hsaKmtQueueSignalExternalSemaphore", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_OPTIONAL, SAGR_PROVIDER_LAYER_EXTERNAL_SEMAPHORE, 1U},
    {"hsaKmtQueueWaitExternalSemaphore", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_OPTIONAL, SAGR_PROVIDER_LAYER_EXTERNAL_SEMAPHORE, 1U},
    {"hsaKmtHandleExport", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtMemoryVaMap", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtMemoryVaUnmap", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtMemHandleFree", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtMemHandleFreePreserveMetadata", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtMemoryGetCpuAddr", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtGetAmdGPUDeviceFd", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtMemoryCpuMap", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL, 1U},
    {"hsaKmtGetNodeWallclockFrequency", SAGR_PROVIDER_SYMBOL_HSA,
     SAGR_PROVIDER_SYMBOL_MANDATORY, SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY,
     1U},
    {"amdgpu_device_deinitialize", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"amdgpu_query_gpu_info", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"amdgpu_bo_cpu_map", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"amdgpu_bo_free", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"amdgpu_bo_export", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"amdgpu_bo_import", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"amdgpu_bo_va_op", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"amdgpu_bo_query_info", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"amdgpu_bo_set_metadata", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
    {"drmCommandWriteRead", SAGR_PROVIDER_SYMBOL_DRM,
     SAGR_PROVIDER_SYMBOL_MANDATORY,
     SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE, 0U},
};

static const provider_layout_record_t k_layouts[] = {
    {"HsaVersionInfo", 8U, 2U},
    {"HsaSystemProperties", 16U, 4U},
    {"HsaNodeProperties", 396U, 10U},
    {"HsaMemoryProperties", 32U, 6U},
    {"HsaCacheProperties", 1056U, 4U},
    {"HsaIoLinkProperties", 52U, 4U},
    {"HsaMemFlags", 4U, 1U},
    {"HsaGraphicsResourceInfo", 40U, 6U},
    {"HsaQueueInfo", 76U, 9U},
    {"HsaQueueResource", 40U, 5U},
    {"HsaEventDescriptor", 24U, 3U},
    {"HsaEvent", 48U, 2U},
    {"HsaPointerInfo", 68U, 11U},
    {"HsaMemoryRange", 16U, 2U},
    {"HsaHandleImportDesc", 40U, 5U},
    {"HsaHandleImportResult", 24U, 4U},
    {"HsaStructureSizes", 16U, 4U},
};

#define LAYOUT_FIELDS(...) {__VA_ARGS__}
static const provider_layout_field_record_t k_layout_fields[][11] = {
    LAYOUT_FIELDS({"KernelInterfaceMajorVersion", 0U},
                  {"KernelInterfaceMinorVersion", 4U}),
    LAYOUT_FIELDS({"NumNodes", 0U}, {"PlatformOem", 4U},
                  {"PlatformId", 8U}, {"PlatformRev", 12U}),
    LAYOUT_FIELDS({"NumCPUCores", 0U}, {"Capability", 32U},
                  {"WaveFrontSize", 52U}, {"LocalMemSize", 92U},
                  {"MarketingName", 112U}, {"AMDName", 240U},
                  {"DebugProperties", 308U}, {"UniqueID", 340U},
                  {"WallClockKHz", 384U}, {"FabricHandleSupported", 392U}),
    LAYOUT_FIELDS({"HeapType", 0U}, {"SizeInBytes", 4U},
                  {"Flags", 12U}, {"Width", 16U},
                  {"MemoryClockMax", 20U}, {"VirtualBaseAddress", 24U}),
    LAYOUT_FIELDS({"ProcessorIdLow", 0U}, {"CacheLevel", 4U},
                  {"CacheType", 28U}, {"SiblingMap", 32U}),
    LAYOUT_FIELDS({"IoLinkType", 0U}, {"NodeFrom", 12U},
                  {"MinimumLatency", 24U}, {"Flags", 48U}),
    LAYOUT_FIELDS({"Value", 0U}),
    LAYOUT_FIELDS({"MemoryAddress", 0U}, {"SizeInBytes", 8U},
                  {"Metadata", 16U}, {"MetadataSizeInBytes", 24U},
                  {"NodeId", 28U}, {"SizeHintInBytes", 32U}),
    LAYOUT_FIELDS({"QueueDetailError", 0U}, {"QueueTypeExtended", 4U},
                  {"NumCUAssigned", 8U}, {"CUMaskInfo", 12U},
                  {"UserContextSaveArea", 20U}, {"SaveAreaSizeInBytes", 28U},
                  {"ControlStackTop", 36U}, {"SaveAreaHeader", 52U},
                  {"SaveAreaAllocSize", 68U}),
    LAYOUT_FIELDS({"QueueId", 0U}, {"Queue_DoorBell", 8U},
                  {"Queue_write_ptr", 16U}, {"Queue_read_ptr", 24U},
                  {"ErrorReason", 32U}),
    LAYOUT_FIELDS({"EventType", 0U}, {"NodeId", 4U}, {"SyncVar", 8U}),
    LAYOUT_FIELDS({"EventId", 0U}, {"EventData", 4U}),
    LAYOUT_FIELDS({"Type", 0U}, {"Node", 4U}, {"MemFlags", 8U},
                  {"CPUAddress", 12U}, {"GPUAddress", 20U},
                  {"SizeInBytes", 28U}, {"NRegisteredNodes", 36U},
                  {"NMappedNodes", 40U}, {"RegisteredNodes", 44U},
                  {"MappedNodes", 52U}, {"UserData", 60U}),
    LAYOUT_FIELDS({"MemoryAddress", 0U}, {"SizeInBytes", 8U}),
    LAYOUT_FIELDS({"device_handle", 0U}, {"type", 8U},
                  {"dmabuf_fd", 12U}, {"mem", 28U}, {"metadata", 36U}),
    LAYOUT_FIELDS({"buf_handle", 0U}, {"dmabuf_fd", 8U},
                  {"alloc_size", 12U}, {"metadata", 20U}),
    LAYOUT_FIELDS({"StructureSizes", 0U}, {"SizeOfHsaNodeProperties", 2U},
                  {"SizeOfHsaExternalHandleDesc", 4U}, {"Reserved", 6U}),
};
#undef LAYOUT_FIELDS

static const provider_model_command_record_t k_model_commands[] = {
    {"HSAKMT_DRM_BO_VA_OP", "struct hsakmt_drm_bo_va_op_args", 0U, 48U},
    {"HSAKMT_DRM_BO_FREE", "struct hsakmt_drm_bo_free_args", 1U, 8U},
    {"HSAKMT_DRM_BO_IMPORT", "struct hsakmt_drm_bo_import_args", 2U, 16U},
    {"HSAKMT_DRM_BO_EXPORT", "struct hsakmt_drm_bo_export_args", 3U, 24U},
    {"HSAKMT_DRM_BO_CPU_MAP", "struct hsakmt_drm_bo_cpu_map_args", 4U, 16U},
    {"HSAKMT_DRM_BO_QUERY_INFO", "struct hsakmt_drm_bo_query_info_args", 5U,
     16U},
    {"HSAKMT_DRM_BO_SET_METADATA", "struct hsakmt_drm_bo_set_metadata_args",
     6U, 16U},
    {"HSAKMT_DRM_COMMAND_WRITE_READ",
     "struct hsakmt_drm_cmd_write_read_args", 7U, 32U},
    {"HSAKMT_DRM_OPEN_RENDER", "struct hsakmt_drm_open_render_args", 8U,
     16U},
    {"HSAKMT_DRM_CLOSE", "struct hsakmt_drm_close_args", 9U, 4U},
    {"HSAKMT_DRM_DEVICE_INITIALIZE",
     "struct hsakmt_drm_device_initialize_args", 10U, 32U},
    {"HSAKMT_DRM_DEVICE_DEINITIALIZE",
     "struct hsakmt_drm_device_deinitialize_args", 11U, 8U},
    {"HSAKMT_DRM_DEVICE_GET_FD", "struct hsakmt_drm_device_get_fd_args", 12U,
     16U},
    {"HSAKMT_DRM_GET_MARKETING_NAME",
     "struct hsakmt_drm_get_marketing_name_args", 13U, 16U},
    {"HSAKMT_DRM_QUERY_GPU_INFO", "struct hsakmt_drm_query_gpu_info_args", 14U,
     16U},
};

_Static_assert(sizeof(k_symbols) / sizeof(k_symbols[0]) ==
                   SAGR_PROVIDER_LOADER_ENTRY_COUNT,
               "provider symbol inventory must match authority");
_Static_assert(sizeof(k_layouts) / sizeof(k_layouts[0]) ==
                   SAGR_PROVIDER_LAYOUT_COUNT,
               "provider layout inventory must match authority");
_Static_assert(sizeof(k_model_commands) / sizeof(k_model_commands[0]) ==
                   SAGR_PROVIDER_MODEL_DRM_COMMAND_COUNT,
               "provider model inventory must match authority");

static const provider_layout_field_record_t *layout_field(
    uint32_t layout_index, uint32_t field_index) {
  if (layout_index >= SAGR_PROVIDER_LAYOUT_COUNT ||
      field_index >= k_layouts[layout_index].field_count) {
    return NULL;
  }
  return &k_layout_fields[layout_index][field_index];
}

static void initialize_error(sagr_error_info_t *error, uint32_t error_size) {
  if (error == NULL || error_size == 0U) {
    return;
  }
  if (error_size < sizeof(*error)) {
    if (error_size >= sizeof(error->struct_size)) {
      error->struct_size = (uint32_t)sizeof(*error);
    }
    return;
  }
  memset(error, 0, sizeof(*error));
  error->struct_size = (uint32_t)sizeof(*error);
}

static void set_error(sagr_error_info_t *error, uint32_t error_size,
                      sagr_status_t status, const char *message) {
  initialize_error(error, error_size);
  if (error != NULL && error_size >= sizeof(*error)) {
    error->status = status;
    if (message != NULL) {
      (void)snprintf(error->message, sizeof(error->message), "%s", message);
    }
  }
}

static int valid_provider(const sagr_provider_t *provider) {
  return provider != NULL && provider->magic == SAGR_PROVIDER_MAGIC &&
         provider->instance != NULL;
}

const char *sagr_provider_authority_sha256(void) {
  return SAGR_PROVIDER_AUTHORITY_SHA256_HEX;
}

const char *sagr_provider_source_commit(void) {
  return SAGR_PROVIDER_SOURCE_COMMIT_HEX;
}

const char *sagr_provider_source_tree(void) {
  return SAGR_PROVIDER_SOURCE_TREE_HEX;
}

const char *sagr_provider_status_string(sagr_provider_hsakmt_status_t status) {
  switch (status) {
    case SAGR_PROVIDER_HSAKMT_STATUS_SUCCESS:
      return "success";
    case SAGR_PROVIDER_HSAKMT_STATUS_ERROR:
      return "error";
    case SAGR_PROVIDER_HSAKMT_STATUS_DRIVER_MISMATCH:
      return "driver mismatch";
    case SAGR_PROVIDER_HSAKMT_STATUS_INVALID_PARAMETER:
      return "invalid parameter";
    case SAGR_PROVIDER_HSAKMT_STATUS_INVALID_HANDLE:
      return "invalid handle";
    case SAGR_PROVIDER_HSAKMT_STATUS_INVALID_NODE_UNIT:
      return "invalid node or unit";
    case SAGR_PROVIDER_HSAKMT_STATUS_NO_MEMORY:
      return "no memory";
    case SAGR_PROVIDER_HSAKMT_STATUS_BUFFER_TOO_SMALL:
      return "buffer too small";
    case SAGR_PROVIDER_HSAKMT_STATUS_NOT_IMPLEMENTED:
      return "not implemented";
    case SAGR_PROVIDER_HSAKMT_STATUS_NOT_SUPPORTED:
      return "not supported";
    case SAGR_PROVIDER_HSAKMT_STATUS_UNAVAILABLE:
      return "unavailable";
    case SAGR_PROVIDER_HSAKMT_STATUS_OUT_OF_RESOURCES:
      return "out of resources";
    case SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED:
      return "kernel channel not opened";
    case SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR:
      return "kernel communication error";
    case SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_ALREADY_OPENED:
      return "kernel already opened";
    case SAGR_PROVIDER_HSAKMT_STATUS_HSAMMU_UNAVAILABLE:
      return "HSAMMU unavailable";
    case SAGR_PROVIDER_HSAKMT_STATUS_WAIT_FAILURE:
      return "wait failure";
    case SAGR_PROVIDER_HSAKMT_STATUS_WAIT_TIMEOUT:
      return "wait timeout";
    case SAGR_PROVIDER_HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED:
      return "memory already registered";
    case SAGR_PROVIDER_HSAKMT_STATUS_MEMORY_NOT_REGISTERED:
      return "memory not registered";
    case SAGR_PROVIDER_HSAKMT_STATUS_MEMORY_ALIGNMENT:
      return "memory alignment";
    default:
      return "unknown status";
  }
}

sagr_status_t sagr_provider_status_to_runtime(
    sagr_provider_hsakmt_status_t status) {
  switch (status) {
    case SAGR_PROVIDER_HSAKMT_STATUS_SUCCESS:
      return SAGR_STATUS_SUCCESS;
    case SAGR_PROVIDER_HSAKMT_STATUS_INVALID_PARAMETER:
      return SAGR_STATUS_INVALID_ARGUMENT;
    case SAGR_PROVIDER_HSAKMT_STATUS_INVALID_HANDLE:
      return SAGR_STATUS_INVALID_HANDLE;
    case SAGR_PROVIDER_HSAKMT_STATUS_NO_MEMORY:
    case SAGR_PROVIDER_HSAKMT_STATUS_OUT_OF_RESOURCES:
      return SAGR_STATUS_OUT_OF_RESOURCES;
    case SAGR_PROVIDER_HSAKMT_STATUS_BUFFER_TOO_SMALL:
      return SAGR_STATUS_BUFFER_TOO_SMALL;
    case SAGR_PROVIDER_HSAKMT_STATUS_NOT_SUPPORTED:
    case SAGR_PROVIDER_HSAKMT_STATUS_NOT_IMPLEMENTED:
      return SAGR_STATUS_NOT_SUPPORTED;
    case SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED:
      return SAGR_STATUS_UNAVAILABLE;
    case SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR:
      return SAGR_STATUS_CONNECTION_LOST;
    case SAGR_PROVIDER_HSAKMT_STATUS_WAIT_TIMEOUT:
      return SAGR_STATUS_TIMED_OUT;
    default:
      return SAGR_STATUS_INTERNAL_ERROR;
  }
}

sagr_status_t sagr_provider_manifest(sagr_provider_manifest_t *manifest,
                                     uint32_t manifest_size) {
  if (manifest == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (manifest_size < sizeof(*manifest)) {
    if (manifest_size >= sizeof(manifest->struct_size)) {
      manifest->struct_size = (uint32_t)sizeof(*manifest);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(manifest, 0, manifest_size);
  manifest->struct_size = manifest_size;
  manifest->loader_entry_count = SAGR_PROVIDER_LOADER_ENTRY_COUNT;
  manifest->mandatory_entry_count = SAGR_PROVIDER_MANDATORY_ENTRY_COUNT;
  manifest->optional_entry_count = SAGR_PROVIDER_OPTIONAL_ENTRY_COUNT;
  manifest->target_loader_entry_count =
      SAGR_PROVIDER_TARGET_LOADER_ENTRY_COUNT;
  manifest->target_mandatory_entry_count =
      SAGR_PROVIDER_TARGET_MANDATORY_ENTRY_COUNT;
  manifest->target_optional_entry_count =
      SAGR_PROVIDER_TARGET_OPTIONAL_ENTRY_COUNT;
  manifest->direct_loader_entry_count = SAGR_PROVIDER_DIRECT_LOADER_ENTRY_COUNT;
  manifest->direct_target_loader_entry_count =
      SAGR_PROVIDER_DIRECT_TARGET_LOADER_ENTRY_COUNT;
  manifest->direct_target_mandatory_entry_count =
      SAGR_PROVIDER_DIRECT_TARGET_MANDATORY_ENTRY_COUNT;
  manifest->direct_target_optional_entry_count =
      SAGR_PROVIDER_DIRECT_TARGET_OPTIONAL_ENTRY_COUNT;
  manifest->hsa_symbol_count = SAGR_PROVIDER_HSA_SYMBOL_COUNT;
  manifest->drm_symbol_count = SAGR_PROVIDER_DRM_SYMBOL_COUNT;
  manifest->target_hsa_symbol_count = SAGR_PROVIDER_TARGET_HSA_SYMBOL_COUNT;
  manifest->target_drm_symbol_count = SAGR_PROVIDER_TARGET_DRM_SYMBOL_COUNT;
  manifest->direct_target_hsa_symbol_count =
      SAGR_PROVIDER_DIRECT_TARGET_HSA_SYMBOL_COUNT;
  manifest->direct_target_drm_symbol_count =
      SAGR_PROVIDER_DIRECT_TARGET_DRM_SYMBOL_COUNT;
  manifest->version_export_count = SAGR_PROVIDER_VERSION_EXPORT_COUNT;
  manifest->layout_count = SAGR_PROVIDER_LAYOUT_COUNT;
  manifest->source_file_count = SAGR_PROVIDER_SOURCE_FILE_COUNT;
  manifest->model_interface_major = SAGR_PROVIDER_MODEL_INTERFACE_MAJOR;
  manifest->model_interface_minor = SAGR_PROVIDER_MODEL_INTERFACE_MINOR;
  manifest->model_function_table_bytes =
      SAGR_PROVIDER_MODEL_FUNCTION_TABLE_BYTES;
  manifest->model_drm_command_count = SAGR_PROVIDER_MODEL_DRM_COMMAND_COUNT;
  manifest->pointer_bytes = 8U;
  manifest->enum_bytes = 4U;
  manifest->packing_bytes = 4U;
  manifest->little_endian = 1U;
  (void)snprintf(manifest->authority_sha256, sizeof(manifest->authority_sha256),
                 "%s", SAGR_PROVIDER_AUTHORITY_SHA256_HEX);
  (void)snprintf(manifest->source_commit, sizeof(manifest->source_commit),
                 "%s", SAGR_PROVIDER_SOURCE_COMMIT_HEX);
  (void)snprintf(manifest->source_tree, sizeof(manifest->source_tree), "%s",
                 SAGR_PROVIDER_SOURCE_TREE_HEX);
  (void)snprintf(manifest->platform, sizeof(manifest->platform), "%s",
                 SAGR_PROVIDER_PLATFORM);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_provider_get_symbol(uint32_t symbol_index,
                                       sagr_provider_symbol_info_t *info,
                                       uint32_t info_size) {
  const provider_symbol_record_t *record;
  if (info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  if (symbol_index >= SAGR_PROVIDER_LOADER_ENTRY_COUNT) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  record = &k_symbols[symbol_index];
  memset(info, 0, info_size);
  info->struct_size = info_size;
  info->index = symbol_index;
  info->kind = record->kind;
  info->requirement = record->requirement;
  info->layer = record->layer;
  info->version_script_exported = record->version_script_exported;
  info->shared_target_effective =
      strcmp(record->name, "hsaKmtGetMemoryHandle") == 0 ? 0U : 1U;
  info->direct_target_effective =
      (strcmp(record->name, "hsaKmtGetMemoryHandle") == 0 ||
       strcmp(record->name, "hsaKmtQueueRingDoorbell") == 0)
          ? 0U
          : 1U;
  (void)snprintf(info->name, sizeof(info->name), "%s", record->name);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_provider_get_layout(uint32_t layout_index,
                                       sagr_provider_layout_info_t *info,
                                       uint32_t info_size) {
  const provider_layout_record_t *record;
  if (info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  if (layout_index >= SAGR_PROVIDER_LAYOUT_COUNT) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  record = &k_layouts[layout_index];
  memset(info, 0, info_size);
  info->struct_size = info_size;
  info->index = layout_index;
  info->size_bytes = record->size_bytes;
  info->field_count = record->field_count;
  (void)snprintf(info->type_name, sizeof(info->type_name), "%s",
                 record->type_name);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_provider_get_layout_field(
    uint32_t layout_index, uint32_t field_index,
    sagr_provider_layout_field_t *field, uint32_t field_size) {
  const provider_layout_field_record_t *record;
  if (field == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (field_size < sizeof(*field)) {
    if (field_size >= sizeof(field->struct_size)) {
      field->struct_size = (uint32_t)sizeof(*field);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  record = layout_field(layout_index, field_index);
  if (record == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(field, 0, field_size);
  field->struct_size = field_size;
  field->layout_index = layout_index;
  field->field_index = field_index;
  field->offset_bytes = record->offset_bytes;
  (void)snprintf(field->field_name, sizeof(field->field_name), "%s",
                 record->name);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_provider_model_manifest(sagr_provider_model_info_t *info,
                                           uint32_t info_size) {
  if (info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(info, 0, info_size);
  info->struct_size = info_size;
  info->interface_major = SAGR_PROVIDER_MODEL_INTERFACE_MAJOR;
  info->interface_minor = SAGR_PROVIDER_MODEL_INTERFACE_MINOR;
  info->function_table_bytes = SAGR_PROVIDER_MODEL_FUNCTION_TABLE_BYTES;
  info->drm_command_count = SAGR_PROVIDER_MODEL_DRM_COMMAND_COUNT;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_provider_model_get_command(
    uint32_t command_index, sagr_provider_model_command_info_t *info,
    uint32_t info_size) {
  const provider_model_command_record_t *record;
  if (info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  if (command_index >= SAGR_PROVIDER_MODEL_DRM_COMMAND_COUNT) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  record = &k_model_commands[command_index];
  memset(info, 0, info_size);
  info->struct_size = info_size;
  info->index = command_index;
  info->value = record->value;
  info->size_bytes = record->size_bytes;
  (void)snprintf(info->name, sizeof(info->name), "%s", record->name);
  (void)snprintf(info->argument_type, sizeof(info->argument_type), "%s",
                 record->argument_type);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_provider_open(
    const char *endpoint, const sagr_instance_open_options_t *options,
    sagr_provider_t **out_provider, sagr_error_info_t *out_error,
    uint32_t error_size) {
  sagr_instance_t instance = NULL;
  sagr_provider_t *provider;
  sagr_status_t status;

  if (out_provider != NULL) {
    *out_provider = NULL;
  }
  if (out_provider == NULL || endpoint == NULL ||
      (out_error == NULL && error_size != 0U)) {
    set_error(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT,
              "invalid provider output, endpoint, or error buffer");
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (out_error != NULL && error_size < sizeof(*out_error)) {
    initialize_error(out_error, error_size);
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }

  status = sagr_instance_open(endpoint, options, &instance, out_error,
                              error_size);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  provider = (sagr_provider_t *)calloc(1, sizeof(*provider));
  if (provider == NULL) {
    (void)sagr_instance_close(&instance);
    set_error(out_error, error_size, SAGR_STATUS_OUT_OF_RESOURCES,
              "could not allocate provider state");
    return SAGR_STATUS_OUT_OF_RESOURCES;
  }
  provider->magic = SAGR_PROVIDER_MAGIC;
  provider->instance = instance;
  atomic_init(&provider->kmt_operation_sequence, UINT64_C(1));
  status = sagr_instance_get_info(instance, &provider->transport_info,
                                  (uint32_t)sizeof(provider->transport_info));
  if (status != SAGR_STATUS_SUCCESS) {
    (void)sagr_instance_close(&provider->instance);
    provider->magic = 0U;
    free(provider);
    set_error(out_error, error_size, status,
              "could not capture provider transport identity");
    return status;
  }
  *out_provider = provider;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_provider_get_info(sagr_provider_t *provider,
                                     sagr_provider_info_t *info,
                                     uint32_t info_size) {
  if (!valid_provider(provider)) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (info == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_STATUS_BUFFER_TOO_SMALL;
  }
  memset(info, 0, info_size);
  info->struct_size = info_size;
  info->state = SAGR_PROVIDER_STATE_OPEN;
  info->connection_id = provider->transport_info.connection_id;
  info->epoch = provider->transport_info.epoch;
  info->rank = provider->transport_info.rank;
  info->world_size = provider->transport_info.world_size;
  memcpy(info->negotiated_capabilities,
         provider->transport_info.negotiated_capabilities,
         sizeof(info->negotiated_capabilities));
  memcpy(info->daemon_uuid, provider->transport_info.daemon_uuid,
         sizeof(info->daemon_uuid));
  memcpy(info->job_uuid, provider->transport_info.job_uuid,
         sizeof(info->job_uuid));
  info->peer_uid = provider->transport_info.peer_uid;
  info->peer_pid = provider->transport_info.peer_pid;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_provider_close(sagr_provider_t **provider) {
  sagr_status_t status;
  if (provider == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (*provider == NULL) {
    return SAGR_STATUS_SUCCESS;
  }
  if (!valid_provider(*provider)) {
    return SAGR_STATUS_INVALID_HANDLE;
  }
  status = sagr_instance_close(&(*provider)->instance);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  (*provider)->magic = 0U;
  free(*provider);
  *provider = NULL;
  return SAGR_STATUS_SUCCESS;
}

sagr_provider_hsakmt_status_t sagr_provider_query_lifecycle(
    sagr_provider_t *provider, uint32_t *is_open) {
  if (!valid_provider(provider)) {
    return SAGR_PROVIDER_HSAKMT_STATUS_INVALID_HANDLE;
  }
  if (is_open == NULL) {
    return SAGR_PROVIDER_HSAKMT_STATUS_INVALID_PARAMETER;
  }
  *is_open = 1U;
  return SAGR_PROVIDER_HSAKMT_STATUS_SUCCESS;
}

sagr_provider_hsakmt_status_t sagr_provider_invoke(
    sagr_provider_t *provider, uint32_t symbol_index, const void *arguments,
    uint32_t argument_size, void *result, uint32_t result_size,
    const sagr_provider_call_options_t *options,
    sagr_error_info_t *out_error, uint32_t error_size) {
  if (!valid_provider(provider)) {
    set_error(out_error, error_size, SAGR_STATUS_INVALID_HANDLE,
              "provider handle is not open");
    return SAGR_PROVIDER_HSAKMT_STATUS_INVALID_HANDLE;
  }
  if (symbol_index >= SAGR_PROVIDER_LOADER_ENTRY_COUNT ||
      (argument_size != 0U && arguments == NULL) ||
      (result_size != 0U && result == NULL)) {
    set_error(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT,
              "invalid provider symbol or argument carrier");
    return SAGR_PROVIDER_HSAKMT_STATUS_INVALID_PARAMETER;
  }
  if (options != NULL && options->struct_size < sizeof(*options)) {
    set_error(out_error, error_size, SAGR_STATUS_INVALID_ARGUMENT,
              "provider call options are too small");
    return SAGR_PROVIDER_HSAKMT_STATUS_INVALID_PARAMETER;
  }
  (void)result;
  (void)result_size;
  set_error(out_error, error_size, SAGR_STATUS_NOT_SUPPORTED,
            "provider symbol is outside the CP-0009 capability boundary");
  return SAGR_PROVIDER_HSAKMT_STATUS_NOT_SUPPORTED;
}
