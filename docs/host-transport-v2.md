# Host Transport v2 Generic Dispatch Boundary

CP-0022 froze the payload-v2 codec for a future generic code-object launch.
CP-0023 accepts the bounded client/adapter/admission layer: an append-only
runtime API and a separate owner-bound native gem5 state machine. CP-0024 adds
an owner-bound type-18 handler source path, type-19 response plumbing, a shared
route-policy harness, and an opt-in live endpoint probe. The authority is
`protocol/host-transport-v2.json`; the implementations are in
`projects/self-amdgpu-runtime` and `projects/gem5`.

This is deliberately an opt-in extension. The envelope remains the existing
80-byte, big-endian, CRC-32C v1 frame, while the payload declares major `2`,
minor `0`. Existing v1 queue, memory, signal, pinned-dispatch, and A1
code-object records are unchanged. The new runtime capability is word 0 bit 8,
`GENERIC_DISPATCH_V2`; on the 32-byte wire bitmap this is byte 1 bit 0,
mask `0x01`. It is valid only with topology, queue, memory, signal, and
code-object transport capabilities. The current daemon does not advertise the
bit. CP-0024's live runtime-to-gem5 probe therefore observes a canonical
unsupported-capability hello and then reconnects with the baseline capability
set. It never sends MessageType 18, so the positive socket route remains
unproven. Codec or handler source presence must not cause advertisement before
the complete session lifecycle exists.

## Records

Each v2 record is exactly 4096 bytes: 80 bytes of the existing envelope and
4016 bytes of payload. Message types 18, 19, and 20 are respectively a generic
dispatch request, acknowledgement, and completion. There are no ancillary
descriptors, host pointers, file descriptors, or client-owned AQL bytes.

The request common prefix is:

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 2 | payload major, `2` |
| 2 | 2 | payload minor, `0` |
| 4 | 2 | opcode: `MAP_OBJECT=1`, `ALLOC_KERNARG=2`, `SUBMIT_AQL=3`, `UNMAP_OBJECT=4` |
| 6 | 2 | flags, `0` |
| 8/16 | 8/8 | object id and generation |
| 24/32 | 8/8 | mapping id and generation |
| 40/48/56 | 8/8/8 | queue id, generation, sequence |
| 64 | 4 | kernel index |
| 68 | 4 | reserved, `0` |
| 72 | 32 | image SHA-256 |
| 104 | 128 | ASCII kernel name, NUL padded |

`MAP_OBJECT` uses offsets 232 through 252 for gfx target 950, relocation count,
kernarg size/alignment, descriptor preload DWORDs, and page size. The initial
gate accepts only zero relocations, 4 KiB or 64 KiB pages, and zero preload.
Nonzero preload is an explicit `NOT_SUPPORTED` result. This is important for
the retained Triton artifact: its descriptor says 12 DWORDs (48 bytes), not
12 bytes, and the existing native command processor has not yet implemented
the corresponding preload and entry-offset semantics.

`ALLOC_KERNARG` uses offsets 232, 240, 248, and 252 for size, alignment, flags,
and reserved data. It binds a nonzero object and mapping identity but does not
accept queue or signal identities. Byte publication remains a separate v1
memory carrier operation; no new KERNARG opcode exists. The CP23 runtime client
exercises a sealed-memfd v1
`MEMORY_COPY_H2D` publication against its mock transport, while the native
selftest calls an owner-bound `publishKernarg` adapter. Those two local tests
are not connected by a proven daemon route. CP24's handler policy accepts only
page-backed ALLOC alignments 4096 and 65536; the normal runtime/Triton logical
alignment 8 remains incompatible and is rejected before mutation. The first
MAP also fixes page size for that owner session.

`SUBMIT_AQL` uses offsets 232 through 332 for the kernarg allocation and range,
signal identity and expected value, grid/workgroup dimensions, Triton launch
metadata, shared memory, wavefront size, and reserved flags. The bounded
validator requires a direct expected signal value of one, wavefront 64, a
workgroup product equal to `num_warps * 64`, grid dimensions at least as large
as their workgroup dimensions, and bounded sizes. It carries no raw packet.
The wire contract requires the daemon to construct the 64-byte AQL packet after
validating ownership. CP23 proves the corresponding operations locally by
materializing, publishing, and fetching a packet before extracted CP-core
admission. CP24's handler rejects SUBMIT_AQL as unsupported because queue,
signal, packet, admission/start/end/retire tick, and type-20 completion
ownership are not yet wired end to end.

`UNMAP_OBJECT` carries only object and mapping identity. All queue, signal,
kernel, hash/name, body, and tail bytes are zero.

Every unused request byte through offset 4015 is zero. Decoding validates the
envelope identity, request correlation, exact frame size, CRC, payload version,
capability dependencies, stage-specific canonical fields, and zero padding.

## Acknowledgements and completion

The response prefix starts with payload version, status, opcode, flags, and
error code. It then echoes owner-scoped object/mapping identities and can return
daemon-issued simulator GPU VAs for mapped segments, the descriptor, code,
entry, kernarg, and packet. It also carries allocation linkage, ticket/trace
IDs, queue/signal identity, packet/output CRCs, and simulator ticks. The
`request_id` and message type remain envelope-derived and are not serialized a
second time.

An OK response is stage-specific: MAP returns mapping identities and mapped
VAs; ALLOC returns a kernarg allocation and VA; SUBMIT returns queue/signal,
kernarg, ticket/trace, and packet linkage; UNMAP returns no resource fields.
For a failed response, all resource identities, VAs, hashes, CRCs, tickets,
traces, and ticks are zero except the echoed opcode, and `error_code` is
nonzero. Noncanonical failures are protocol errors rather than partial state.

The intended live ordering remains:

`MAP_OBJECT -> ALLOC_KERNARG -> v1 MEMORY_COPY_H2D -> SUBMIT_AQL -> completion -> UNMAP_OBJECT`

The mapping, allocation, queue, signal, and object generations must belong to
the same connection and remain pinned through completion. A public
`simulated_va` value is not automatically packet-visible GPU memory; only a
daemon-issued mapping/lease response from the routed handler can establish
that relation.

## CP23 accepted local boundary

The runtime child preserves the v1 ABI and passes its complete CTest suite
18/18. Its v2 client test covers MAP, ALLOC, sealed-memfd kernarg publication,
SUBMIT, failed and successful completion, wait, UNMAP, stale/cross-owner
rejection, and no-capability failure against an in-process mock transport.

The gem5 protocol codec passes 47/47 normally and 47/47 under ASAN/UBSAN. A
separate no-x86 native selftest binds one owner and request epoch across MAP,
ALLOC_KERNARG, `publishKernarg`, queue publish/fetch, extracted command-
processor admission, retire, and UNMAP. If the CP rejects after fetch,
`cancelFetch` restores the slot to the queue owner before mapped/kernarg pins
are released; a retry or explicit cancellation cannot silently lose the slot.

These are two local halves, not an end-to-end daemon path. Capability bit 8 is
not advertised, MessageType 18 is not routed, and admission does not reach
GPUDispatcher, Shader, ComputeUnit, or kernel execution. Normal Triton launcher,
compiler/JIT, output correctness, and fallback remain false.

## CP24 accepted bounded partial boundary

Gem5 now contains a per-client, owner/generation-bound generic session and a
MessageType 18 dispatch branch that can encode canonical MessageType 19
responses. Disconnect cleanup precedes legacy memory and code-object cleanup.
The shared route-policy harness passes absent-capability canonical failure and
no-mutation checks, MAP/UNMAP policy, 4K/64K page policy, alignment-8 rejection,
and fail-closed SUBMIT. This proves handler and policy source behavior, not a
positive production socket route.

The runtime suite has 20 tests: 19 pass and the endpoint test is the one
expected skip when its environment variable is unset. A separate retained run
points that test at a live VEGA_X86 gem5 listener. The required-generic hello is
canonically rejected because bit 8 is not advertised, then a baseline hello
succeeds and closes cleanly. No type-18 frame or v1 H2D publication is sent in
that live run.

Accordingly `handler_source_present` and `route_policy_harness` are true, while
`daemon_socket_route`, `h2d_daemon_route`, `submit_ack`,
`generic_completion_type20`, launcher, compiler/JIT, execution, and fallback
remain false. The v1 `MEMORY_COPY_H2D` record remains the sole kernarg byte
carrier; the v2 opcode set is unchanged.

## Triton boundary and non-claims

The CP-0021 retained Triton tutorial and HSACO identities are recorded in the
authority JSON. The codec can represent its 48-byte kernarg and 24832-by-256
work-item/workgroup geometry, but the 12-DWORD (48-byte) descriptor preload is
rejected until preload-aware mapping and entry semantics exist. CP24 continues
to return NOT_SUPPORTED instead of stripping or reinterpreting that field.

The next implementation checkpoint is CP-0025 /
`P5-TRITON-VECADD-04-DAEMON-LIFECYCLE`. It must reconcile logical alignment 8
with the page-backed native allocation contract, bind v1 MEMORY_COPY_H2D to the
same owner/session, and implement queue/signal ownership, daemon-built AQL,
packet CRC, nonzero admission/start/end/retire ticks, SUBMIT ACK, and type-20
completion. Only after a live positive lifecycle and disconnect/cancellation
tests pass may capability bit 8 be reconsidered. GPU execution and the normal
Triton launcher remain later gates unless separately proven.
