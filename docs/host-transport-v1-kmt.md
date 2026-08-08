# KFD/DRM Operation Envelope

This document explains the CP-0010 authority in
[`protocol/host-transport-v1-kmt.json`](../protocol/host-transport-v1-kmt.json).
It is the wire contract for the typed `libhsakmt` shim and a GemSim daemon
callback boundary. It is not a claim that a host KFD or production ROCr stack
is available.

## Fixed record

`KMT_OPERATION_V1` uses capability bit 5 and message types 14 (`KMT_REQUEST`)
and 15 (`KMT_ACK`). Both records have the base 80-byte header and a fixed
256-byte big-endian payload, for 336 bytes per record. There are no ancillary
descriptors and no fragmentation. The header request ID remains the transport
correlation key; the payload operation sequence is an owner-scoped monotonic
sequence for typed calls.

The request has fixed-width owner/object/aux ID and generation pairs, eight
32-bit argument words, and one 128-byte copied buffer. The result mirrors those
identities and carries a 128-byte copied result. Bytes beyond the declared
length are zero and the declared region is checked with CRC-32C. A caller
therefore passes values and copied bytes, never a host pointer, pointer-sized
handle, FD, dma-buf, or SCM_RIGHTS descriptor.

## Operations and model callbacks

The 18 stable IDs cover lifecycle/version, topology snapshot, allocation/free/
copy, queue create/destroy/doorbell, event create/destroy/set/reset/query/wait,
pointer-info, and model DRM calls needed by the typed shim. `MODEL_DRM_CALL` selects one of the
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

This checkpoint freezes the envelope and its testable invariants. It does not
prove a complete 124-symbol provider, KFD attach, memory/queue/event behavior,
HIP/OpenCL registration, Triton/PyTorch/vLLM execution, or Qwen inference.
Those claims require the typed child implementation and independent runtime
evidence in later gates.
