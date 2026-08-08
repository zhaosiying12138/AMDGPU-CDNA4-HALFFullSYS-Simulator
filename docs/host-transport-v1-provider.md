# ROCr/libhsakmt Provider Authority

This document explains the machine-readable authority in
[`protocol/host-transport-v1-provider.json`](../protocol/host-transport-v1-provider.json).
It is the CP-0009 source-inventory boundary. It does not claim that a GemSim
provider, HIP, OpenCL, or a generic code-object loader already exists.

## Frozen source

The source lane is `rocm-systems` `develop` at commit
`92115a2941982a384de161be3f78cf9bff547027`, tree
`28bf42b65f7aad25167180543dda69b5fc6caf58`. The relevant subtrees are
`rocr-runtime` tree `e68c0afe1ce52ab5ec3582bcbde95a5681d44f98`, its HSA runtime
tree `bd8d06a20f7a41a3001659fd4ae7f565c251171f` and `core` subtree
`3c01bd8a2d12f5de12da83741fa27482daeec1d8`, and `libhsakmt` tree
`3a70cc0c6401ff26f18273c80f2015cc5df81936`.

The authority records the Git blob and SHA-256 for each source file. The
important files are:

| File | Source role |
| --- | --- |
| `runtime/hsa-runtime/core/inc/thunk_loader.h` | 113 HSA KMT PFN typedefs, 11 DRM typedefs, and the loader table |
| `runtime/hsa-runtime/core/runtime/thunk_loader.cpp` | selection, loading, symbol resolution, direct binding, DTIF lifecycle, and DXG ABI check |
| `libhsakmt/include/hsakmt/hsakmt.h` | public C declarations and the documented Linux device path |
| `libhsakmt/include/hsakmt/hsakmttypes.h` | status enum and `#pragma pack(..., 4)` ABI structures |
| `libhsakmt/include/hsakmt/hsakmtmodeliface.h` | software-model function table, DRM command enum, and argument layouts |
| `libhsakmt/src/libhsakmt.ver` | Linux shared-object export list |
| `libhsakmt/src/openclose.c` | KFD open/close and kernel-channel status behavior |
| `libhsakmt/src/memory.c`, `fmm.c`, `debug.c` | upstream hardware call sites retained as an exclusion audit |

The recorded 17 HSA layouts and the model/DRM layouts are for the pinned Linux x86-64 ABI: eight-byte pointers,
four-byte enums, little-endian integers, and the header's four-byte packing.
For the HSA records, `field_scope=key_offsets_partial`: each `fields` map is a
selected set of key offsets, not an exhaustive list of upstream fields; the
recorded `sizeof` remains authoritative. They are not a license to copy the
numbers to another ABI. A provider build must rerun the layout probe and fail
on any mismatch.

## Loader behavior

`ThunkLoader::whoami()` has three source-defined outcomes
(`thunk_loader.cpp:64-89`):

1. `enable_dtif` selects `libdtif.so` (`dtif64a.dll` on Windows).
2. On Linux, `enable_dxg_detection` probes `/dev/dxg`; a successful probe
   selects `librocdxg.so`.
3. Otherwise the library name is empty and the table is bound directly to the
   linked `hsaKmt*` and `amdgpu_*` functions.

In a shared-library branch the constructor tries `LoadLib(library_name)` and
then the library adjacent to the loader image. `LoadThunkApiTable()` then calls
`GetExportAddress` in the exact 124-entry order in the JSON authority. There
are 119 mandatory entries. A missing mandatory entry jumps to `LOAD_ERROR`; it
does not create a partially usable table. The five optional entries are
`hsaKmtSetSigbusDelay` and the four external-semaphore functions. A missing
optional export leaves a null PFN for a guarded call.

The direct branch assigns PFNs to linked symbols in its own source order,
recorded as `direct_order` in JSON. The source-union table has 124 entries.
On Linux, shared resolution has 123 active entries (HSA 112 + DRM 11), while
direct binding has 122 (HSA 111 + DRM 11): `hsaKmtGetMemoryHandle` is `_WIN32`
only in both branches and `hsaKmtQueueRingDoorbell` is additionally guarded
in direct binding. The syntactic table remains 124 entries so the audit cannot
silently lose a platform-specific symbol.

After loading, DTIF optionally resolves `DtifCreate`/`DtifDestroy`. DXG
optionally resolves `DxgAbiCheck`; an old library without that export is
treated as ABI-compatible, otherwise `HsaStructureSizes` must be accepted.
GemSim must never enter these hardware branches.

## Symbol layers

The 113 source-union HSA PFNs are partitioned into lifecycle/topology, event synchronization,
queue/dispatch, memory/virtual address, debug/observability, and optional
external-semaphore layers. The 11 DRM entries form a separate hardware layer;
the seven JSON layers cover all 124 source-union entries exactly once.
The version script exports 108 HSA names. The five PFN names absent from that
script are `hsaKmtCreateQueueExt`, `hsaKmtCreateQueueV2`,
`hsaKmtModelEnabled`, `hsaKmtQueueRingDoorbell`, and
`hsaKmtRegisterGraphicsHandleToNodesExt`; this difference is intentional and
is a required audit result, not permission to omit a provider symbol.

The authority records the public names and signatures for a future full
provider. The current CP-0009 implementation is a metadata-only skeleton: it
publishes the manifest, symbol/layout/model metadata, and a generic unsupported
invoke boundary, but it does not export the 124 typed hsaKmt/DRM entry points.
The future provider gate requires that complete typed surface, including
deterministic `HSAKMT_STATUS_NOT_SUPPORTED` (or the explicitly documented
`NOT_IMPLEMENTED` policy) after argument and ownership validation. A caller
must be able to distinguish an invalid argument or stale handle from a
capability that is not implemented.

## Lifecycle query boundary

`query_lifecycle` is deliberately scoped to `transport-open`: it reports that
a valid provider handle completed the host-transport handshake and remains
open. It does not open KFD, acquire KFD topology, discover a GPU node, or imply
that a KFD channel is open. The query performs no `/dev/kfd`, `/dev/dri`, DRM,
or topology operation. Those are separate provider operations and remain
subject to their own implementation or unsupported-status gates.

## Status boundary

The source enum has 21 assigned values, including `SUCCESS=0`,
`INVALID_PARAMETER=3`, `INVALID_HANDLE=4`, `NO_MEMORY=6`,
`NOT_IMPLEMENTED=10`, `NOT_SUPPORTED=11`, `OUT_OF_RESOURCES=13`,
`KERNEL_IO_CHANNEL_NOT_OPENED=20`, `KERNEL_COMMUNICATION_ERROR=21`,
`WAIT_TIMEOUT=31`, and the memory registration/alignment statuses 35-37.
The full table is in JSON.

The existing KfdDriver does not apply one universal conversion: most generic
non-success values become `HSA_STATUS_ERROR`, allocation failure becomes
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`, and external semaphore wrappers preserve
invalid-argument, invalid-agent, and not-supported distinctions. Those exact
source observations are recorded with file/line references. The future full
provider policy uses `KERNEL_COMMUNICATION_ERROR` for a failed daemon
transaction and reserves `KERNEL_IO_CHANNEL_NOT_OPENED` for a KFD channel that
has not been opened. CP-0009's `transport-open` lifecycle query does not use
that status; it only reports the transport handshake state.

`SUCCESS` is allowed only after the corresponding daemon-side state transition
has committed. Unsupported work returns `NOT_SUPPORTED` (or explicitly
documented `NOT_IMPLEMENTED`) and cannot mutate allocations, queues, events,
signals, or output bytes.

## No hardware device path

The provider is a host-side daemon client. It must not open, stat, probe, or
load:

* `/dev/kfd`
* `/dev/dri`
* `/dev/dxg`
* `/dev/udmabuf`
* production `libhsakmt`, `libdrm`, AMDGPU UMD, or `amdsmi` libraries

The upstream source documents `/dev/kfd` in `hsakmt.h` and opens it in
`openclose.c`; that is evidence for the semantic ABI only, not an allowed
GemSim implementation dependency. The provider gate therefore requires a
syscall audit, direct-call/dependency audit, and negative device-node test in addition to
the ordinary lifecycle, memory, queue, event, ownership, OOM, double-free,
concurrency, and DSO tests.

The model interface is a source-defined 1.1 function table with 15 DRM command
argument structs. `handle_drm_call` is only an upstream model hook; it does not
virtualize every production libhsakmt path. `memory.c`, `fmm.c`, and `debug.c`
still contain direct or wrapper calls to libdrm/AMDGPU and render-node probes.
GemSim must keep those calls out of its dependency graph and return
`HSAKMT_STATUS_NOT_SUPPORTED` until a daemon-owned equivalent is specified.

## Acceptance boundary

This authority proves only that CP-0009 has a source-exact ABI inventory and a
mechanical implementation contract. It does not prove provider code, HIP or
OpenCL registration, Triton execution, arbitrary HSACO/ELF loading, model
execution, or performance. Those claims require later checkpoints and their
own retained evidence.
