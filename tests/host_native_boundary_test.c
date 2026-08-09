/* SPDX-License-Identifier: GPL-3.0-or-later */

/*
 * CP-0014-A: this is an intentionally small consumer of the installed public
 * ABI.  It must remain runnable on the physical host without a gem5 process,
 * ROCr loader, or GPU device.  Transport-backed behavior is covered by the
 * existing protocol tests; this test freezes the host-native seam itself.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <self_amdgpu_runtime/code_object.h>
#include <self_amdgpu_runtime/kmt_shim.h>
#include <self_amdgpu_runtime/provider.h>
#include <self_amdgpu_runtime/runtime.h>

static int expect(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "host-native boundary: %s\n", message);
    return 1;
  }
  return 0;
}

int main(void) {
  int failures = 0;
  sagr_instance_open_options_t instance_options;
  sagr_queue_create_options_t queue_options;
  sagr_queue_operation_options_t queue_operation_options;
  sagr_memory_allocate_options_t memory_options;
  sagr_memory_operation_options_t memory_operation_options;
  sagr_signal_create_options_t signal_options;
  sagr_signal_operation_options_t signal_operation_options;
  sagr_pinned_dispatch_options_t dispatch_options;
  sagr_kmt_call_options_t kmt_options;
  sagr_kmt_envelope_request_t kmt_request;
  sagr_kmt_envelope_result_t kmt_result;
  sagr_provider_manifest_t provider_manifest;
  sagr_provider_model_info_t model_info;
  sagr_code_object_info_t code_object_info;
  sagr_kmt_handle_t kfd_handle;
  sagr_instance_t instance = NULL;
  sagr_provider_t *provider = NULL;
  uint8_t byte = 0;

  /* These payloads are fixed-width carriers, not host pointers. */
  _Static_assert(sizeof(sagr_kmt_handle_t) == 32,
                 "KMT handle ABI changed");
  _Static_assert(sizeof(sagr_kmt_envelope_request_t) == SAGR_KMT_PAYLOAD_BYTES,
                 "KMT request ABI changed");
  _Static_assert(sizeof(sagr_kmt_envelope_result_t) == SAGR_KMT_PAYLOAD_BYTES,
                 "KMT result ABI changed");

  failures += expect(sagr_abi_version() == SAGR_ABI_VERSION,
                     "runtime ABI version mismatch");
  failures += expect(sagr_instance_open_options_init(
                         &instance_options,
                         (uint32_t)sizeof(instance_options)) ==
                         SAGR_STATUS_SUCCESS,
                     "instance options are not host-initializable");
  failures += expect(sagr_queue_create_options_init(
                         &queue_options, (uint32_t)sizeof(queue_options)) ==
                         SAGR_STATUS_SUCCESS,
                     "queue options are not host-initializable");
  failures += expect(sagr_queue_operation_options_init(
                         &queue_operation_options,
                         (uint32_t)sizeof(queue_operation_options)) ==
                         SAGR_STATUS_SUCCESS,
                     "queue operation options are not host-initializable");
  failures += expect(sagr_memory_allocate_options_init(
                         &memory_options, (uint32_t)sizeof(memory_options)) ==
                         SAGR_STATUS_SUCCESS,
                     "memory options are not host-initializable");
  failures += expect(sagr_memory_operation_options_init(
                         &memory_operation_options,
                         (uint32_t)sizeof(memory_operation_options)) ==
                         SAGR_STATUS_SUCCESS,
                     "memory operation options are not host-initializable");
  failures += expect(sagr_signal_create_options_init(
                         &signal_options, (uint32_t)sizeof(signal_options)) ==
                         SAGR_STATUS_SUCCESS,
                     "signal options are not host-initializable");
  failures += expect(sagr_signal_operation_options_init(
                         &signal_operation_options,
                         (uint32_t)sizeof(signal_operation_options)) ==
                         SAGR_STATUS_SUCCESS,
                     "signal operation options are not host-initializable");
  failures += expect(sagr_pinned_dispatch_options_init(
                         &dispatch_options, (uint32_t)sizeof(dispatch_options)) ==
                         SAGR_STATUS_SUCCESS,
                     "dispatch options are not host-initializable");
  failures += expect(sagr_kmt_call_options_init(
                         &kmt_options, (uint32_t)sizeof(kmt_options)) ==
                         SAGR_KMT_STATUS_SUCCESS,
                     "KMT options are not host-initializable");
  failures += expect(sagr_kmt_envelope_request_init(
                         &kmt_request, (uint32_t)sizeof(kmt_request),
                         SAGR_KMT_OP_OPEN_KFD) == SAGR_KMT_STATUS_SUCCESS,
                     "KMT request carrier is not host-initializable");

  /* Metadata and code-object validation are local operations with no device. */
  failures += expect(sagr_provider_manifest(
                         &provider_manifest,
                         (uint32_t)sizeof(provider_manifest)) ==
                         SAGR_STATUS_SUCCESS,
                     "provider manifest requires a device");
  failures += expect(sagr_provider_model_manifest(
                         &model_info, (uint32_t)sizeof(model_info)) ==
                         SAGR_STATUS_SUCCESS,
                     "provider model manifest requires a device");
  failures += expect(provider_manifest.model_interface_major ==
                         SAGR_PROVIDER_MODEL_INTERFACE_MAJOR,
                     "provider manifest/model boundary drifted");
  failures += expect(model_info.interface_major ==
                         SAGR_PROVIDER_MODEL_INTERFACE_MAJOR,
                     "model interface version drifted");
  failures += expect(sagr_code_object_validate(
                         NULL, 0, &code_object_info,
                         (uint32_t)sizeof(code_object_info)) ==
                         SAGR_STATUS_INVALID_ARGUMENT,
                     "code-object validation accepted a null host image");
  failures += expect(sagr_code_object_gemsim_isa_supported(NULL) == 0,
                     "code-object ISA query accepted a null image");

  /* Invalid handles fail locally; no endpoint or host GPU is touched. */
  failures += expect(sagr_provider_open(
                         NULL, &instance_options, &provider, NULL, 0) ==
                         SAGR_STATUS_INVALID_ARGUMENT &&
                         provider == NULL,
                     "provider open did not reject a null endpoint locally");
  failures += expect(sagr_provider_close(&provider) == SAGR_STATUS_SUCCESS &&
                         provider == NULL,
                     "provider close was not null-safe");
  failures += expect(sagr_kmt_open_kfd(
                         NULL, &kfd_handle, &kmt_options, NULL, 0) ==
                         SAGR_KMT_STATUS_INVALID_HANDLE,
                     "KMT open did not reject a null provider locally");
  failures += expect(sagr_memory_copy_from_host(
                         NULL, 0, &byte, 1, &memory_operation_options, NULL,
                         0) == SAGR_STATUS_INVALID_HANDLE,
                     "memory copy did not reject a null handle locally");
  failures += expect(sagr_instance_close(&instance) == SAGR_STATUS_SUCCESS &&
                         instance == NULL,
                     "instance close did not preserve null ownership");

  /* A zero result is intentionally rejected before any transport exchange. */
  memset(&kmt_result, 0, sizeof(kmt_result));
  failures += expect(sagr_kmt_envelope_result_validate(
                         &kmt_request, &kmt_result) ==
                         SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                     "KMT result validation crossed the host boundary");

  return failures == 0 ? 0 : 1;
}
