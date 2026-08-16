/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_PROVIDER_H
#define SELF_AMDGPU_RUNTIME_PROVIDER_H

#include <stdint.h>

#include <self_amdgpu_runtime/export.h>
#include <self_amdgpu_runtime/runtime.h>
#include <self_amdgpu_runtime/kmt_shim.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This header is the small CP-0009 provider boundary.  The strings and
 * counts are copied from protocol/host-transport-v1-provider.json; they are
 * identity data, not a claim that the corresponding production API is linked
 * into this library.
 */
#define SAGR_PROVIDER_AUTHORITY_SCHEMA \
  "amdgpu-sim.host-transport-v1.provider.v1"
#define SAGR_PROVIDER_AUTHORITY_SHA256_HEX \
  "12c66e523a2f91750379255c6021c34d457432a3b818062b69f8077595a124bc"
#define SAGR_PROVIDER_SOURCE_COMMIT_HEX \
  "92115a2941982a384de161be3f78cf9bff547027"
#define SAGR_PROVIDER_SOURCE_TREE_HEX \
  "28bf42b65f7aad25167180543dda69b5fc6caf58"
#define SAGR_PROVIDER_PLATFORM "linux-x86_64"

#define SAGR_PROVIDER_LOADER_ENTRY_COUNT UINT32_C(124)
#define SAGR_PROVIDER_MANDATORY_ENTRY_COUNT UINT32_C(119)
#define SAGR_PROVIDER_OPTIONAL_ENTRY_COUNT UINT32_C(5)
#define SAGR_PROVIDER_TARGET_LOADER_ENTRY_COUNT UINT32_C(123)
#define SAGR_PROVIDER_TARGET_MANDATORY_ENTRY_COUNT UINT32_C(118)
#define SAGR_PROVIDER_TARGET_OPTIONAL_ENTRY_COUNT UINT32_C(5)
#define SAGR_PROVIDER_DIRECT_LOADER_ENTRY_COUNT UINT32_C(124)
#define SAGR_PROVIDER_DIRECT_TARGET_LOADER_ENTRY_COUNT UINT32_C(122)
#define SAGR_PROVIDER_DIRECT_TARGET_MANDATORY_ENTRY_COUNT UINT32_C(117)
#define SAGR_PROVIDER_DIRECT_TARGET_OPTIONAL_ENTRY_COUNT UINT32_C(5)
#define SAGR_PROVIDER_HSA_SYMBOL_COUNT UINT32_C(113)
#define SAGR_PROVIDER_DRM_SYMBOL_COUNT UINT32_C(11)
#define SAGR_PROVIDER_TARGET_HSA_SYMBOL_COUNT UINT32_C(112)
#define SAGR_PROVIDER_TARGET_DRM_SYMBOL_COUNT UINT32_C(11)
#define SAGR_PROVIDER_DIRECT_TARGET_HSA_SYMBOL_COUNT UINT32_C(111)
#define SAGR_PROVIDER_DIRECT_TARGET_DRM_SYMBOL_COUNT UINT32_C(11)
#define SAGR_PROVIDER_VERSION_EXPORT_COUNT UINT32_C(108)
#define SAGR_PROVIDER_LAYOUT_COUNT UINT32_C(17)
#define SAGR_PROVIDER_SOURCE_FILE_COUNT UINT32_C(18)
#define SAGR_PROVIDER_MODEL_INTERFACE_MAJOR UINT32_C(1)
#define SAGR_PROVIDER_MODEL_INTERFACE_MINOR UINT32_C(1)
#define SAGR_PROVIDER_MODEL_FUNCTION_TABLE_BYTES UINT32_C(32)
#define SAGR_PROVIDER_MODEL_DRM_COMMAND_COUNT UINT32_C(15)
#define SAGR_PROVIDER_SYMBOL_NAME_BYTES UINT32_C(64)
#define SAGR_PROVIDER_LAYOUT_NAME_BYTES UINT32_C(48)
#define SAGR_PROVIDER_LAYOUT_FIELD_NAME_BYTES UINT32_C(48)

typedef struct sagr_provider sagr_provider_t;

/* The source HSAKMT status enum is a four-byte integer on the target ABI. */
typedef int32_t sagr_provider_hsakmt_status_t;
enum {
  SAGR_PROVIDER_HSAKMT_STATUS_SUCCESS = 0,
  SAGR_PROVIDER_HSAKMT_STATUS_ERROR = 1,
  SAGR_PROVIDER_HSAKMT_STATUS_DRIVER_MISMATCH = 2,
  SAGR_PROVIDER_HSAKMT_STATUS_INVALID_PARAMETER = 3,
  SAGR_PROVIDER_HSAKMT_STATUS_INVALID_HANDLE = 4,
  SAGR_PROVIDER_HSAKMT_STATUS_INVALID_NODE_UNIT = 5,
  SAGR_PROVIDER_HSAKMT_STATUS_NO_MEMORY = 6,
  SAGR_PROVIDER_HSAKMT_STATUS_BUFFER_TOO_SMALL = 7,
  SAGR_PROVIDER_HSAKMT_STATUS_NOT_IMPLEMENTED = 10,
  SAGR_PROVIDER_HSAKMT_STATUS_NOT_SUPPORTED = 11,
  SAGR_PROVIDER_HSAKMT_STATUS_UNAVAILABLE = 12,
  SAGR_PROVIDER_HSAKMT_STATUS_OUT_OF_RESOURCES = 13,
  SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED = 20,
  SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR = 21,
  SAGR_PROVIDER_HSAKMT_STATUS_KERNEL_ALREADY_OPENED = 22,
  SAGR_PROVIDER_HSAKMT_STATUS_HSAMMU_UNAVAILABLE = 23,
  SAGR_PROVIDER_HSAKMT_STATUS_WAIT_FAILURE = 30,
  SAGR_PROVIDER_HSAKMT_STATUS_WAIT_TIMEOUT = 31,
  SAGR_PROVIDER_HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED = 35,
  SAGR_PROVIDER_HSAKMT_STATUS_MEMORY_NOT_REGISTERED = 36,
  SAGR_PROVIDER_HSAKMT_STATUS_MEMORY_ALIGNMENT = 37
};

typedef enum sagr_provider_symbol_kind {
  SAGR_PROVIDER_SYMBOL_HSA = 1,
  SAGR_PROVIDER_SYMBOL_DRM = 2
} sagr_provider_symbol_kind_t;

typedef enum sagr_provider_symbol_requirement {
  SAGR_PROVIDER_SYMBOL_MANDATORY = 1,
  SAGR_PROVIDER_SYMBOL_OPTIONAL = 2
} sagr_provider_symbol_requirement_t;

typedef enum sagr_provider_symbol_layer {
  SAGR_PROVIDER_LAYER_UNCLASSIFIED = 0,
  SAGR_PROVIDER_LAYER_LIFECYCLE_TOPOLOGY = 1,
  SAGR_PROVIDER_LAYER_EVENT_SYNC = 2,
  SAGR_PROVIDER_LAYER_QUEUE_DISPATCH = 3,
  SAGR_PROVIDER_LAYER_MEMORY_VIRTUAL = 4,
  SAGR_PROVIDER_LAYER_DEBUG_AND_OBSERVABILITY = 5,
  SAGR_PROVIDER_LAYER_EXTERNAL_SEMAPHORE = 6,
  SAGR_PROVIDER_LAYER_DRM_HARDWARE_SURFACE = 7
} sagr_provider_symbol_layer_t;

typedef struct sagr_provider_symbol_info {
  uint32_t struct_size;
  uint32_t index;
  uint32_t kind;
  uint32_t requirement;
  uint32_t layer;
  uint32_t version_script_exported;
  uint32_t shared_target_effective;
  uint32_t direct_target_effective;
  char name[SAGR_PROVIDER_SYMBOL_NAME_BYTES];
} sagr_provider_symbol_info_t;

typedef struct sagr_provider_layout_info {
  uint32_t struct_size;
  uint32_t index;
  uint32_t size_bytes;
  /* Number of key offsets recorded by the authority, not a field total. */
  uint32_t field_count;
  char type_name[SAGR_PROVIDER_LAYOUT_NAME_BYTES];
} sagr_provider_layout_info_t;

/* Layout fields are the authority's recorded key-offset map; omitted fields
 * are intentionally not inferred and do not imply a complete struct ABI. */
typedef struct sagr_provider_layout_field {
  uint32_t struct_size;
  uint32_t layout_index;
  uint32_t field_index;
  uint32_t offset_bytes;
  char field_name[SAGR_PROVIDER_LAYOUT_FIELD_NAME_BYTES];
} sagr_provider_layout_field_t;

typedef struct sagr_provider_manifest {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t loader_entry_count;
  uint32_t mandatory_entry_count;
  uint32_t optional_entry_count;
  uint32_t target_loader_entry_count;
  uint32_t target_mandatory_entry_count;
  uint32_t target_optional_entry_count;
  uint32_t direct_loader_entry_count;
  uint32_t direct_target_loader_entry_count;
  uint32_t direct_target_mandatory_entry_count;
  uint32_t direct_target_optional_entry_count;
  uint32_t hsa_symbol_count;
  uint32_t drm_symbol_count;
  uint32_t target_hsa_symbol_count;
  uint32_t target_drm_symbol_count;
  uint32_t direct_target_hsa_symbol_count;
  uint32_t direct_target_drm_symbol_count;
  uint32_t version_export_count;
  uint32_t layout_count;
  uint32_t source_file_count;
  uint32_t model_interface_major;
  uint32_t model_interface_minor;
  uint32_t model_function_table_bytes;
  uint32_t model_drm_command_count;
  uint32_t pointer_bytes;
  uint32_t enum_bytes;
  uint32_t packing_bytes;
  uint8_t little_endian;
  uint8_t reserved0[3];
  char authority_sha256[65];
  char source_commit[41];
  char source_tree[41];
  char platform[32];
  uint8_t reserved[16];
} sagr_provider_manifest_t;

typedef struct sagr_provider_model_info {
  uint32_t struct_size;
  uint32_t interface_major;
  uint32_t interface_minor;
  uint32_t function_table_bytes;
  uint32_t drm_command_count;
  uint32_t reserved[3];
} sagr_provider_model_info_t;

typedef struct sagr_provider_model_command_info {
  uint32_t struct_size;
  uint32_t index;
  uint32_t value;
  uint32_t size_bytes;
  char name[SAGR_PROVIDER_SYMBOL_NAME_BYTES];
  char argument_type[SAGR_PROVIDER_SYMBOL_NAME_BYTES];
} sagr_provider_model_command_info_t;

typedef enum sagr_provider_state {
  SAGR_PROVIDER_STATE_CLOSED = 0,
  SAGR_PROVIDER_STATE_OPEN = 1
} sagr_provider_state_t;

typedef struct sagr_provider_info {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t state;
  uint32_t reserved0;
  uint64_t connection_id;
  uint64_t epoch;
  uint32_t rank;
  uint32_t world_size;
  uint64_t negotiated_capabilities[SAGR_CAPABILITY_WORD_COUNT];
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t job_uuid[SAGR_UUID_SIZE];
  uint32_t peer_uid;
  uint32_t peer_pid;
  uint8_t reserved[32];
} sagr_provider_info_t;

/* A call is metadata-only at CP-0009.  No raw pointer is serialized. */
typedef struct sagr_provider_call_options {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t reserved[4];
} sagr_provider_call_options_t;

SAGR_API const char *sagr_provider_authority_sha256(void);
SAGR_API const char *sagr_provider_source_commit(void);
SAGR_API const char *sagr_provider_source_tree(void);
SAGR_API const char *sagr_provider_status_string(
    sagr_provider_hsakmt_status_t status);
SAGR_API sagr_status_t sagr_provider_status_to_runtime(
    sagr_provider_hsakmt_status_t status);

SAGR_API sagr_status_t sagr_provider_manifest(
    sagr_provider_manifest_t *manifest, uint32_t manifest_size);
SAGR_API sagr_status_t sagr_provider_get_symbol(
    uint32_t symbol_index, sagr_provider_symbol_info_t *info,
    uint32_t info_size);
SAGR_API sagr_status_t sagr_provider_get_layout(
    uint32_t layout_index, sagr_provider_layout_info_t *info,
    uint32_t info_size);
SAGR_API sagr_status_t sagr_provider_get_layout_field(
    uint32_t layout_index, uint32_t field_index,
    sagr_provider_layout_field_t *field, uint32_t field_size);
SAGR_API sagr_status_t sagr_provider_model_manifest(
    sagr_provider_model_info_t *info, uint32_t info_size);
SAGR_API sagr_status_t sagr_provider_model_get_command(
    uint32_t command_index, sagr_provider_model_command_info_t *info,
    uint32_t info_size);

/*
 * Opening the provider performs exactly the existing transport handshake. It
 * never selects a host device or production runtime. The provider owns the
 * returned transport instance and close releases it.
 */
SAGR_API sagr_status_t sagr_provider_open(
    const char *endpoint, const sagr_instance_open_options_t *options,
    sagr_provider_t **out_provider, sagr_error_info_t *out_error,
    uint32_t error_size);
/*
 * Managed opens retain the same provider contract while assigning simulator
 * process, registry, and transport ownership to one managed session.  They
 * explicitly require the generic KMT capability; ordinary managed-runtime
 * callers keep their existing capability set and lifecycle.
 */
SAGR_API sagr_status_t sagr_provider_open_managed(
    const sagr_managed_session_options_t *options,
    sagr_provider_t **out_provider, sagr_managed_session_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_provider_open_managed_v2(
    const sagr_managed_session_options_v2_t *options,
    sagr_provider_t **out_provider, sagr_managed_session_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_provider_get_info(
    sagr_provider_t *provider, sagr_provider_info_t *info, uint32_t info_size);
SAGR_API sagr_status_t sagr_provider_close(sagr_provider_t **provider);
/*
 * A provider is process-owned and every normal operation fails after fork.
 * A child that does not immediately exec may discard its inherited local copy
 * without sending remote teardown or terminating the parent-owned simulator.
 * Calling this on the creating process is rejected.
 */
SAGR_API sagr_status_t sagr_provider_discard_inherited(
    sagr_provider_t **provider);

/* Source lifecycle/query hooks. They are deliberately semantic, not KFD I/O. */
SAGR_API sagr_provider_hsakmt_status_t sagr_provider_query_lifecycle(
    sagr_provider_t *provider, uint32_t *is_open);

/*
 * Invoke validates ownership and argument carriers, then reports the current
 * capability boundary. It never writes result or changes provider state for a
 * valid but unsupported symbol.
 */
SAGR_API sagr_provider_hsakmt_status_t sagr_provider_invoke(
    sagr_provider_t *provider, uint32_t symbol_index,
    const void *arguments, uint32_t argument_size, void *result,
    uint32_t result_size, const sagr_provider_call_options_t *options,
    sagr_error_info_t *out_error, uint32_t error_size);

#ifdef __cplusplus
}
#endif

#endif
