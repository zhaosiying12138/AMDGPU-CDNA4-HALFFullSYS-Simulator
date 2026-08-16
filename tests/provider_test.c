/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <self_amdgpu_runtime/provider.h>

static int expect(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "provider test: %s\n", message);
    return 1;
  }
  return 0;
}

static int check_manifest(void) {
  sagr_provider_manifest_t manifest;
  int failures = 0;
  _Static_assert(sizeof(SAGR_PROVIDER_AUTHORITY_SHA256_HEX) == 65U,
                 "authority SHA-256 literal must contain 64 hex bytes");
  memset(&manifest, 0, sizeof(manifest));
  failures += expect(
      sagr_provider_manifest(&manifest, (uint32_t)sizeof(manifest)) ==
          SAGR_STATUS_SUCCESS,
      "manifest query succeeds");
  failures += expect(manifest.loader_entry_count == 124U,
                     "source-union loader count");
  failures += expect(manifest.mandatory_entry_count == 119U,
                     "source-union mandatory count");
  failures += expect(manifest.optional_entry_count == 5U,
                     "source-union optional count");
  failures += expect(manifest.target_loader_entry_count == 123U,
                     "shared linux effective count");
  failures += expect(manifest.target_mandatory_entry_count == 118U,
                     "shared linux effective mandatory count");
  failures += expect(manifest.direct_loader_entry_count == 124U,
                     "direct source-union count");
  failures += expect(manifest.direct_target_loader_entry_count == 122U,
                     "direct linux effective count");
  failures += expect(manifest.direct_target_mandatory_entry_count == 117U,
                     "direct linux effective mandatory count");
  failures += expect(manifest.hsa_symbol_count == 113U &&
                         manifest.drm_symbol_count == 11U,
                     "source symbol counts");
  failures += expect(manifest.target_hsa_symbol_count == 112U &&
                         manifest.target_drm_symbol_count == 11U,
                     "shared target symbol counts");
  failures += expect(manifest.direct_target_hsa_symbol_count == 111U &&
                         manifest.direct_target_drm_symbol_count == 11U,
                     "direct target symbol counts");
  failures += expect(manifest.version_export_count == 108U &&
                         manifest.layout_count == 17U &&
                         manifest.source_file_count == 18U,
                     "authority inventory counts");
  failures += expect(manifest.model_interface_major == 1U &&
                         manifest.model_interface_minor == 1U &&
                         manifest.model_function_table_bytes == 32U &&
                         manifest.model_drm_command_count == 15U,
                     "model ABI counts");
  failures += expect(manifest.pointer_bytes == 8U && manifest.enum_bytes == 4U &&
                         manifest.packing_bytes == 4U &&
                         manifest.little_endian == 1U,
                     "target ABI properties");
  failures += expect(strcmp(manifest.authority_sha256,
                            SAGR_PROVIDER_AUTHORITY_SHA256_HEX) == 0,
                     "authority hash");
  failures += expect(strlen(sagr_provider_authority_sha256()) == 64U,
                     "authority hash length");
  failures += expect(strcmp(manifest.source_commit,
                            SAGR_PROVIDER_SOURCE_COMMIT_HEX) == 0,
                     "source commit");
  return failures;
}

static int check_symbols(void) {
  static const char *const expected[] = {
      "hsaKmtOpenKFD", "hsaKmtCloseKFD", "hsaKmtGetVersion",
      "hsaKmtAcquireSystemProperties", "hsaKmtReleaseSystemProperties",
      "hsaKmtGetNodeProperties", "hsaKmtGetNodeMemoryProperties",
      "hsaKmtGetNodeCacheProperties", "hsaKmtGetNodeIoLinkProperties",
      "hsaKmtCreateEvent", "hsaKmtDestroyEvent", "hsaKmtSetEvent",
      "hsaKmtResetEvent", "hsaKmtQueryEventState", "hsaKmtWaitOnEvent",
      "hsaKmtWaitOnMultipleEvents", "hsaKmtCreateQueue",
      "hsaKmtCreateQueueExt", "hsaKmtCreateQueueV2", "hsaKmtUpdateQueue",
      "hsaKmtDestroyQueue", "hsaKmtSetQueueCUMask", "hsaKmtSetMemoryPolicy",
      "hsaKmtAllocMemory", "hsaKmtAllocMemoryAlign", "hsaKmtFreeMemory",
      "hsaKmtAvailableMemory", "hsaKmtRegisterMemory",
      "hsaKmtRegisterMemoryToNodes", "hsaKmtRegisterMemoryWithFlags",
      "hsaKmtRegisterGraphicsHandleToNodes",
      "hsaKmtRegisterGraphicsHandleToNodesExt", "hsaKmtShareMemory",
      "hsaKmtRegisterSharedHandle", "hsaKmtRegisterSharedHandleToNodes",
      "hsaKmtProcessVMRead", "hsaKmtProcessVMWrite",
      "hsaKmtDeregisterMemory", "hsaKmtMapMemoryToGPU",
      "hsaKmtMapMemoryToGPUNodes", "hsaKmtUnmapMemoryToGPU",
      "hsaKmtDbgRegister", "hsaKmtDbgUnregister",
      "hsaKmtDbgWavefrontControl", "hsaKmtDbgAddressWatch",
      "hsaKmtDbgEnable", "hsaKmtDbgDisable", "hsaKmtDbgGetDeviceData",
      "hsaKmtDbgGetQueueData", "hsaKmtGetClockCounters",
      "hsaKmtPmcGetCounterProperties", "hsaKmtPmcRegisterTrace",
      "hsaKmtPmcUnregisterTrace", "hsaKmtPmcAcquireTraceAccess",
      "hsaKmtPmcReleaseTraceAccess", "hsaKmtPmcStartTrace",
      "hsaKmtPmcQueryTrace", "hsaKmtPmcStopTrace",
      "hsaKmtMapGraphicHandle", "hsaKmtUnmapGraphicHandle",
      "hsaKmtSetTrapHandler", "hsaKmtSetSigbusDelay", "hsaKmtGetTileConfig",
      "hsaKmtQueryPointerInfo", "hsaKmtSetMemoryUserData",
      "hsaKmtGetQueueInfo", "hsaKmtGetKernelQueueId",
      "hsaKmtAllocQueueGWS", "hsaKmtRuntimeEnable",
      "hsaKmtRuntimeDisable", "hsaKmtCheckRuntimeDebugSupport",
      "hsaKmtGetRuntimeCapabilities", "hsaKmtGetCoreRuntimeInfo",
      "hsaKmtGetCoreDeviceInfo", "hsaKmtDebugTrapIoctl", "hsaKmtSPMAcquire",
      "hsaKmtSPMRelease", "hsaKmtSPMSetDestBuffer", "hsaKmtSVMSetAttr",
      "hsaKmtSVMGetAttr", "hsaKmtSetXNACKMode", "hsaKmtGetXNACKMode",
      "hsaKmtOpenSMI", "hsaKmtExportDMABufHandle", "hsaKmtWaitOnEvent_Ext",
      "hsaKmtWaitOnMultipleEvents_Ext", "hsaKmtReplaceAsanHeaderPage",
      "hsaKmtReturnAsanHeaderPage", "hsaKmtGetAMDGPUDeviceHandle",
      "hsaKmtPcSamplingQueryCapabilities", "hsaKmtPcSamplingCreate",
      "hsaKmtPcSamplingDestroy", "hsaKmtPcSamplingStart",
      "hsaKmtPcSamplingStop", "hsaKmtPcSamplingSupport", "hsaKmtModelEnabled",
      "hsaKmtQueueRingDoorbell", "amdgpu_device_initialize",
      "hsaKmtAisReadWriteFile", "hsaKmtGetMemoryHandle", "hsaKmtHandleImport",
      "hsaKmtImportExternalSemaphore", "hsaKmtDestroyExternalSemaphore",
      "hsaKmtQueueSignalExternalSemaphore", "hsaKmtQueueWaitExternalSemaphore",
      "hsaKmtHandleExport", "hsaKmtMemoryVaMap", "hsaKmtMemoryVaUnmap",
      "hsaKmtMemHandleFree", "hsaKmtMemHandleFreePreserveMetadata",
      "hsaKmtMemoryGetCpuAddr", "hsaKmtGetAmdGPUDeviceFd", "hsaKmtMemoryCpuMap",
      "hsaKmtGetNodeWallclockFrequency", "amdgpu_device_deinitialize",
      "amdgpu_query_gpu_info", "amdgpu_bo_cpu_map", "amdgpu_bo_free",
      "amdgpu_bo_export", "amdgpu_bo_import", "amdgpu_bo_va_op",
      "amdgpu_bo_query_info", "amdgpu_bo_set_metadata", "drmCommandWriteRead",
  };
  int failures = 0;
  size_t index;
  failures += expect(sizeof(expected) / sizeof(expected[0]) == 124U,
                     "test symbol table has 124 entries");
  for (index = 0; index < sizeof(expected) / sizeof(expected[0]); ++index) {
    sagr_provider_symbol_info_t info;
    memset(&info, 0, sizeof(info));
    failures += expect(
        sagr_provider_get_symbol((uint32_t)index, &info, (uint32_t)sizeof(info)) ==
            SAGR_STATUS_SUCCESS,
        "symbol query succeeds");
    failures += expect(strcmp(info.name, expected[index]) == 0,
                       "symbol order matches authority");
    failures += expect(info.index == (uint32_t)index,
                       "symbol index is stable");
  }
  {
    sagr_provider_symbol_info_t info;
    memset(&info, 0, sizeof(info));
    failures += expect(sagr_provider_get_symbol(96U, &info, (uint32_t)sizeof(info)) ==
                           SAGR_STATUS_SUCCESS &&
                           info.shared_target_effective == 1U &&
                           info.direct_target_effective == 0U,
                       "queue doorbell platform guard");
    failures += expect(sagr_provider_get_symbol(99U, &info, (uint32_t)sizeof(info)) ==
                           SAGR_STATUS_SUCCESS &&
                           info.shared_target_effective == 0U &&
                           info.direct_target_effective == 0U,
                       "memory handle platform guard");
  }
  failures += expect(sagr_provider_get_symbol(124U, NULL, 0U) ==
                         SAGR_STATUS_INVALID_ARGUMENT,
                     "out-of-range symbol is rejected");
  return failures;
}

static int check_layouts(void) {
  /* These are recorded key offsets, not an exhaustive upstream field map. */
  static const uint32_t sizes[] = {8U, 16U, 396U, 32U, 1056U, 52U,
                                   4U, 40U, 76U, 40U, 24U, 48U,
                                   68U, 16U, 40U, 24U, 16U};
  static const char *const names[] = {
      "HsaVersionInfo", "HsaSystemProperties", "HsaNodeProperties",
      "HsaMemoryProperties", "HsaCacheProperties", "HsaIoLinkProperties",
      "HsaMemFlags", "HsaGraphicsResourceInfo", "HsaQueueInfo",
      "HsaQueueResource", "HsaEventDescriptor", "HsaEvent",
      "HsaPointerInfo", "HsaMemoryRange", "HsaHandleImportDesc",
      "HsaHandleImportResult", "HsaStructureSizes"};
  int failures = 0;
  size_t index;
  for (index = 0; index < sizeof(sizes) / sizeof(sizes[0]); ++index) {
    sagr_provider_layout_info_t info;
    memset(&info, 0, sizeof(info));
    failures += expect(sagr_provider_get_layout((uint32_t)index, &info,
                                                 (uint32_t)sizeof(info)) ==
                           SAGR_STATUS_SUCCESS,
                       "layout query succeeds");
    failures += expect(info.size_bytes == sizes[index] &&
                           strcmp(info.type_name, names[index]) == 0,
                       "layout identity matches authority");
  }
  {
    sagr_provider_layout_field_t field;
    memset(&field, 0, sizeof(field));
    failures += expect(sagr_provider_get_layout_field(
                           2U, 8U, &field, (uint32_t)sizeof(field)) ==
                           SAGR_STATUS_SUCCESS &&
                           field.offset_bytes == 384U &&
                           strcmp(field.field_name, "WallClockKHz") == 0,
                       "layout field offset matches authority");
  }
  failures += expect(sagr_provider_get_layout(17U, NULL, 0U) ==
                         SAGR_STATUS_INVALID_ARGUMENT,
                     "out-of-range layout is rejected");
  return failures;
}

static int check_model(void) {
  sagr_provider_model_info_t model;
  sagr_provider_model_command_info_t command;
  int failures = 0;
  memset(&model, 0, sizeof(model));
  memset(&command, 0, sizeof(command));
  failures += expect(sagr_provider_model_manifest(&model, (uint32_t)sizeof(model)) ==
                         SAGR_STATUS_SUCCESS &&
                         model.interface_major == 1U && model.interface_minor == 1U &&
                         model.function_table_bytes == 32U &&
                         model.drm_command_count == 15U,
                     "model ABI manifest");
  failures += expect(sagr_provider_model_get_command(
                         0U, &command, (uint32_t)sizeof(command)) ==
                         SAGR_STATUS_SUCCESS && command.value == 0U &&
                         command.size_bytes == 48U &&
                         strcmp(command.name, "HSAKMT_DRM_BO_VA_OP") == 0,
                     "first model command");
  failures += expect(sagr_provider_model_get_command(
                         14U, &command, (uint32_t)sizeof(command)) ==
                         SAGR_STATUS_SUCCESS && command.value == 14U &&
                         command.size_bytes == 16U &&
                         strcmp(command.name, "HSAKMT_DRM_QUERY_GPU_INFO") == 0,
                     "last model command");
  return failures;
}

static int check_status_and_unsupported(void) {
  uint8_t result[32];
  sagr_error_info_t error;
  sagr_managed_session_options_t managed_options;
  sagr_managed_session_options_v2_t managed_options_v2;
  sagr_provider_t *managed_provider = NULL;
  int failures = 0;
  memset(result, 0xa5, sizeof(result));
  memset(&error, 0, sizeof(error));
  failures += expect(sagr_provider_status_to_runtime(
                         SAGR_PROVIDER_HSAKMT_STATUS_NOT_SUPPORTED) ==
                         SAGR_STATUS_NOT_SUPPORTED,
                     "unsupported status mapping");
  failures += expect(sagr_provider_status_to_runtime(
                         SAGR_PROVIDER_HSAKMT_STATUS_INVALID_HANDLE) ==
                         SAGR_STATUS_INVALID_HANDLE,
                     "invalid-handle status mapping");
  failures += expect(strcmp(sagr_provider_status_string(
                                SAGR_PROVIDER_HSAKMT_STATUS_NOT_SUPPORTED),
                            "not supported") == 0,
                     "status string");
  failures += expect(sagr_provider_invoke(
                         NULL, 0U, NULL, 0U, result, (uint32_t)sizeof(result),
                         NULL, &error, (uint32_t)sizeof(error)) ==
                         SAGR_PROVIDER_HSAKMT_STATUS_INVALID_HANDLE,
                     "null provider is rejected");
  failures += expect(error.status == SAGR_STATUS_INVALID_HANDLE,
                     "invoke error status is explicit");
  failures += expect(result[0] == 0xa5U && result[31] == 0xa5U,
                     "invalid invoke does not mutate result");
  {
    sagr_provider_t *provider = NULL;
    sagr_status_t status = sagr_provider_open(
        "/tmp/amdgpu-sim-provider-no-such-endpoint", NULL, &provider, &error,
        (uint32_t)sizeof(error));
    failures += expect(status != SAGR_STATUS_SUCCESS && provider == NULL,
                       "provider open does not fall back to a host device");
  }
  failures += expect(
      sagr_provider_open_managed_v2(
          NULL, &managed_provider, NULL, 0U, &error,
          (uint32_t)sizeof(error)) == SAGR_STATUS_INVALID_ARGUMENT &&
          managed_provider == NULL,
      "managed v2 provider requires exact options");
  failures += expect(
      sagr_managed_session_options_init(
          &managed_options, (uint32_t)sizeof(managed_options)) ==
      SAGR_STATUS_SUCCESS,
      "managed provider options initialize");
  managed_options.flags = UINT32_C(2);
  failures += expect(
      sagr_provider_open_managed(
          &managed_options, &managed_provider, NULL, 0U, &error,
          (uint32_t)sizeof(error)) == SAGR_STATUS_INVALID_ARGUMENT &&
          managed_provider == NULL,
      "managed provider rejects unknown v1 flags before launch");
  failures += expect(
      sagr_managed_session_options_v2_init(
          &managed_options_v2, (uint32_t)sizeof(managed_options_v2)) ==
      SAGR_STATUS_SUCCESS,
      "managed v2 provider options initialize");
  managed_options_v2.struct_size =
      (uint32_t)sizeof(managed_options_v2) - 1U;
  failures += expect(
      sagr_provider_open_managed_v2(
          &managed_options_v2, &managed_provider, NULL, 0U, &error,
          (uint32_t)sizeof(error)) == SAGR_STATUS_INVALID_ARGUMENT &&
          managed_provider == NULL,
      "managed provider validates v2 size before copying");
  return failures;
}

int main(void) {
  int failures = 0;
  _Static_assert(sizeof(sagr_provider_hsakmt_status_t) == 4,
                 "provider status ABI width changed");
  _Static_assert(sizeof(sagr_provider_symbol_info_t) == 96,
                 "provider symbol ABI size changed");
  _Static_assert(sizeof(sagr_provider_layout_info_t) == 64,
                 "provider layout ABI size changed");
  _Static_assert(sizeof(sagr_provider_layout_field_t) == 64,
                 "provider layout field ABI size changed");
  failures += check_manifest();
  failures += check_symbols();
  failures += check_layouts();
  failures += check_model();
  failures += check_status_and_unsupported();
  if (failures != 0) {
    fprintf(stderr, "provider test failures: %d\n", failures);
    return 1;
  }
  puts("provider tests passed");
  return 0;
}
