# KFD/DRM Operation Envelope

This document explains the version-5 CP-0010 authority in
[`protocol/host-transport-v1-kmt.json`](../protocol/host-transport-v1-kmt.json).
It is the wire contract for the typed `libhsakmt` shim and a GemSim daemon
callback boundary. It is not a claim that a host KFD or production ROCr stack
is available.

## Fixed record

`KMT_OPERATION_V1` uses capability bit 5 and message types 14 (`KMT_REQUEST`)
and 15 (`KMT_ACK`). Both records have the base 80-byte header and a fixed
256-byte big-endian payload, for 336 bytes per record. There is no
fragmentation. `EXPORT_BACKING` alone carries exactly one writable shared
memfd through `SCM_RIGHTS`; every other operation rejects ancillary
descriptors. The header request ID remains the transport correlation key; the
payload operation sequence is an owner-scoped monotonic sequence for typed
calls.

The request has fixed-width owner/object/aux ID and generation pairs, eight
32-bit argument words, and one 128-byte copied buffer. The result mirrors those
identities and carries a 128-byte copied result. Bytes beyond the declared
length are zero and the declared region is checked with CRC-32C. A caller
therefore passes values and copied bytes, never a host pointer, pointer-sized
handle, numeric FD, dma-buf, or device descriptor. The one `EXPORT_BACKING`
descriptor is transported out of band, validated, and never encoded as an
integer in either payload.

## Operations and model callbacks

The first 26 stable IDs cover lifecycle/version, topology snapshot, allocation/free/
copy, queue create/destroy/doorbell, event create/destroy/set/reset/query/wait,
pointer-info, model DRM calls, process-aperture paging, and VM acquisition
needed by the typed shim, plus GPU-specific allocation and mapping. Operation
27, `EXPORT_BACKING`, is the project-owned bridge operation that installs the
one owner-scoped shared memfd; it is not an upstream `libhsakmt` symbol.
Operation 28, `GET_CLOCK_COUNTERS`, translates the standard
`AMDKFD_IOC_GET_CLOCK_COUNTERS` request. It returns one coherent nonzero
simulated nanosecond sample for GPU, CPU, and system plus a 1 GHz system-clock
frequency in the eight existing result words; it carries no copied bytes or
descriptor.
`MODEL_DRM_CALL` selects one of the
15 source-defined model command IDs from the CP-0009 authority. The operation
ID is not an upstream ioctl number and does not authorize forwarding an ioctl.

The JSON mechanically assigns each operation's `arg_words` and `result_words`,
handle requirements, allowed flags, copied-buffer direction and size, source
symbol, and result layout. Unlisted words are zero. Paired high/low words use
the documented high-word then low-word convention; this convention is
independent of the enclosing big-endian byte encoding.

KFD operations select the daemon's KFD callback table; operation ID 18 selects
`DRM_CALL`, and argument word zero selects one of its 15 commands. The
source-defined model version is fixed at `1.1`. The daemon dispatches those enums through
its own static callback table. A callback address, `dlopen` result, library
name, pointer, or host descriptor is never serialized. A model version other
than exactly `1.1` is `UNSUPPORTED_VERSION` until a later envelope authority
records a compatibility rule.

Envelope minor `1.5` includes operation 19 for paged
`AMDKFD_IOC_GET_PROCESS_APERTURES_NEW` records and operation 20 for
`AMDKFD_IOC_ACQUIRE_VM`, plus operation 21 for
`AMDKFD_IOC_SET_MEMORY_POLICY`. The former copies fixed 56-byte value records;
the latter two carry only scalar identities and aperture values. The upstream
output pointer and render FD remain inside the Model DSO process and never
cross the runtime-gem5 wire. Operation 26,
`AMDKFD_IOC_SET_SCRATCH_BACKING_VA`, carries the logical GPU and a
page-aligned virtual address as scalar words. It records an owner-scoped
scratch binding and never serializes a host pointer, allocation, or FD.

Operations 22 through 25 add the standard GPU-specific allocation lifecycle:
`AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`, `FREE_MEMORY_OF_GPU`, `MAP_MEMORY_TO_GPU`,
and `UNMAP_MEMORY_FROM_GPU`. GPU VA, size, backing token, flags, and GPU IDs are
scalar or copied fixed-width values; no caller pointer or device ID array
pointer crosses the bridge. Before a file-backed allocation, `EXPORT_BACKING`
installs exactly one owner-scoped memfd. Its ordinary allocation region is
`[0, backing_bytes - 8192)`; the final 8192 bytes are reserved as a generic
doorbell aperture with 128 aligned 8-byte slots. A `DOORBELL` allocation must
cover that complete tail and a `USERPTR` allocation remains an overloaded CPU
address token rather than a file offset. The facade allocates ordinary backing
offsets, the daemon validates every range independently, and neither side
derives a file offset from a simulated GPU VA. Mapping is idempotent per
allocation and is bounded by the 16-GPU logical topology.

On successful `QUEUE_CREATE`, result word 0 returns the accepted queue depth
and result words 1-2 return the queue's absolute byte offset in the exported
memfd. Upstream ROCr may map the containing page and use the within-page
remainder as its doorbell address without any simulator-specific API.

`OPEN_KFD` means creation of a daemon-owned virtual KFD session. It does not
open `/dev/kfd`, inspect `/dev/dri`, discover host topology, or perform a
production DRM operation. A daemon without an implementation returns
`wire_status=OK` plus source `HSAKMT_STATUS_NOT_SUPPORTED` after validation.

## Status and atomicity

The result contains two status domains. `wire_status` is the base transport
status and is `-1` before a result is decoded; `status` preserves the pinned
source HSAKMT status for an admitted operation. A caller checks `wire_status`
first. Only `wire_status=OK` and `status=HSAKMT_STATUS_SUCCESS` can publish a state transition or
result bytes.

Validation precedence is intentionally strict: framing and CRC errors are
dropped, then payload errors, version, identity, topology, capability,
ownership, handle generation, operation arguments, resource limits, provider
status, and finally internal callback/transport failure. A validated
unsupported operation returns `NOT_SUPPORTED` without creating or changing
any owner, handle, generation, allocation, queue, event, model, or output
bytes. A failed create likewise publishes no newly allocated identity.

Malformed records do not advance the operation sequence. A complete,
structurally valid request consumes its sequence exactly once even if the
provider returns an error; sequence exhaustion never wraps. Disconnect
invalidates all owner-scoped state and never replays a request.

## Deterministic fixture

The first daemon test fixture is KFD 1.0 with model ABI 1.1 and one node:
node ID 1, gfx target code 950, one CU, wavefront 64, 64-KiB pages, 48-bit VA,
eight queues, 1024 allocations, and 1024 events. The topology generation is
the negotiated base-header `job_epoch`; it is not a separately invented
counter. The JSON freezes the exact 16-byte version record, 64-byte topology
record for fixture epoch 1, object ID prefixes, generations, and CRC-32C values.

## Boundary

This checkpoint freezes the envelope and its testable invariants. KFD open,
version, process-aperture discovery, owner-scoped VM binding, shared backing,
GPU allocation/mapping, and an unchanged upstream ROCr queue create/destroy
lifecycle are implemented and covered by focused tests. The queue smoke proves
that upstream ROCr accepts the exported backing and returned doorbell offset;
it does not yet prove ring observation, doorbell-triggered AQL consumption,
packet retirement, a complete 124-symbol provider, HIP/OpenCL, Triton,
PyTorch/vLLM/SGLang execution, or model inference. Those claims require
independent runtime evidence in later gates.
