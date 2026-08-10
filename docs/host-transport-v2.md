# Host Transport v2 Generic Dispatch Boundary

CP-0022 froze the payload-v2 codec for a future generic code-object launch.
CP-0023 accepts the bounded client/adapter/admission layer: an append-only
runtime API and a separate owner-bound native gem5 state machine. CP-0024 adds
an owner-bound type-18 handler source path, type-19 response plumbing, a shared
route-policy harness, and an opt-in live endpoint probe. CP-0025 completes the
positive negotiated daemon control lifecycle through native admission and
retirement. CP-0026 adds a separate locked execution extension and a
trace-authoritative disconnect quarantine boundary. CP-0027 reuses that wire
unchanged for the first direct OpenCL executable and a second exact execution
variant. The authority is
`protocol/host-transport-v2.json`; the implementations are in
`projects/self-amdgpu-runtime` and `projects/gem5`.

This is deliberately an opt-in extension. The envelope remains the existing
80-byte, big-endian, CRC-32C v1 frame, while the payload declares major `2`,
minor `0`. Existing v1 queue, memory, signal, pinned-dispatch, and A1
code-object records are unchanged. Runtime word 0 bit 8,
`GENERIC_DISPATCH_V2`, remains the control/admission/retire contract; on the
32-byte wire bitmap this is byte 1 bit 0, mask `0x01`. CP-0026 adds
`GENERIC_EXECUTION_V2` at word 0 bit 9, wire byte 1 bit 1, mask `0x02`. Bit 9
is valid only when bit 8 and topology, queue, memory, signal, and code-object
transport capabilities are selected. Bit-8-only type-20 responses retain
`output_crc32c=0`; bit 9 requires a nonzero output CRC and a matching daemon
trace/output oracle. Capability advertisement alone never proves execution.
The current live route is the `VEGA_X86` bridge; the CP-0020 no-x86 functional
fixture is a separate boundary.

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
selftest calls an owner-bound `publishKernarg` adapter. CP25 connects those
halves through the live daemon. The wire value 8 is the logical kernel ABI
alignment; the allocator privately uses the owner session's 4096- or
65536-byte page backing and returns the logical alignment in the ACK. The first
MAP still fixes page size for that owner session. The retained positive fixture
allocates 512 bytes and publishes the complete 280-byte manifest at offset 64.

`SUBMIT_AQL` uses offsets 232 through 332 for the kernarg allocation and range,
signal identity and expected value, grid/workgroup dimensions, Triton launch
metadata, shared memory, wavefront size, and reserved flags. The bounded
validator requires a direct expected signal value of one, wavefront 64, a
workgroup product equal to `num_warps * 64`, grid dimensions at least as large
as their workgroup dimensions, and bounded sizes. It carries no raw packet.
The wire contract requires the daemon to construct the 64-byte AQL packet after
validating ownership. CP23 proves the corresponding operations locally by
materializing, publishing, and fetching a packet before extracted CP-core
admission. CP25's handler validates the queue and signal owners, constructs and
publishes the packet, binds its nonzero CRC, performs native CP admission,
durably sends the type-19 ACK, and only then emits type-20 retirement. Admission
is nonzero; start, end, and retire are nondecreasing and may share the
bookkeeping tick. No raw client packet or GPU execution claim is introduced.

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

## CP25 accepted positive control boundary

The clean final gem5 and runtime children pass a fresh required-generic live
run with two sequential owner generations. Generation one disconnects with
live leases; the daemon reclaims its owner-scoped resources, clears the active
session latch, and accepts generation two. Generation two completes MAP,
logical-align-8 ALLOC over hidden page backing, v1 `MEMORY_COPY_H2D` at offset
64 for the full 280-byte manifest, SUBMIT, type-19 ACK, type-20 retirement, and
UNMAP. Slot and VA reuse with new generations verifies targeted cleanup;
`remote_resource_counters_observed` remains false, while the focused native
state test separately reports `zero_resources=true`.

The packet CRC and admission tick are nonzero, and
`admission <= start <= end <= retire`. Completion is scheduled only after the
ACK has been durably committed. These timestamps are daemon bookkeeping for
native CP admission and retirement: `gpu_dispatcher`, `compute_unit`,
`kernel_executed`, `execution`, and `output_correctness` all remain false.

## CP26 accepted locked execution boundary

CP-0026 keeps the CP25 control contract and selects bit 9 only as its exact
execution extension. The accepted live fixture is the 5,528-byte gfx950
`gpuReadWrite` HSACO, SHA-256
`7b6a4d2bb7f9c4e7466bcf69f3110ecbfab54d07abd4c70b6bd96b6a6fb9de56`, with
zero descriptor preload. The daemon validates a 4 KiB map, 512-byte
alignment-8 kernarg allocation, a 280-byte OLD_ABI manifest at offset 64, and
three 4 KiB A/B/C allocations before issuing the packet. The live route is
`VEGA_X86` `host_dispatch.py`, while the separate CP-0020 no-x86 B2 fixture is
used as an additional functional differential; neither result is a claim of a
standalone no-x86 daemon.

The positive daemon trace has three ordered events:
`generic_execution_retired`, `generic_execution_type20_durable`, and
`generic_execution_session_complete`. It records one packet fetch and one
GPUDispatcher start, four workgroups, sixteen wave64 waves, 19 instruction
starts per wave (304 total), exact PC coverage, 1,024 global reads, 2,048
global writes, 32 store events, and 2,048 store lanes. The A/B/C oracle is A
unchanged, B=`gid`, C=A; A and C CRC-32C are `0x4705cdab`, B is
`0xb28d0486`, and the B-then-C output CRC is `0x6f67026f`. The runtime endpoint
verifies all three D2H buffers, a duplicate D2H read, and UNMAP. The daemon
trace, not endpoint assertion booleans, establishes GPUDispatcher,
ComputeUnit, execution, and output correctness.

The wire response's `signal_value_bits` remains the expected value `1`, and
the endpoint records `signal_after_observed=false`. Trace
`signal_before=1`/`signal_after=0` is the private native AQL completion signal,
not a wire post-value. A positive completed disconnect is distinct from a
post-ACK quarantine: the negative endpoint closes after durable type-19 and
does not wait, read type-20, D2H, or UNMAP; the daemon emits one
`generic_execution_quarantine_cleanup` event with
`owner_disconnected=true`, `owner_quarantined=true`,
`cleanup_complete=true`, and `type20_durable=false`. Client JSON is not
cleanup authority. Pre-SUBMIT live-lease cleanup and normal completed-session
cleanup remain separate positive boundaries.

## CP27 accepted direct OpenCL boundary

CP-0027 does not add an opcode or reinterpret bit 9. It adds a second
server-side exact execution profile plus a local OpenCL/runtime supervisor. A
normal host executable calls the bounded `libOpenCL.so.1`; `clBuildProgram`
invokes only the versioned repository-local Clang/device-libs and produces the
exact 5,160-byte gfx950 `vecadd` image, SHA-256
`314ede16940432996c9fe190115408bf42744a8ab7d0036bf07b931e39c4cb19`. The same
process creates a private job identity/socket, starts gem5, performs the
existing object/memory/queue/signal/generic records, waits for type 20, reads C,
unmaps, closes, and joins the daemon. The user does not launch an endpoint or
construct transport records.

The variant locks an 88-byte kernarg, zero descriptor preload, 1D global size
1,024, local size 256, and A/B/C 4 KiB owner-bound allocations. The fsynced
trace proves one packet/CP/dispatcher submission, four workgroups, sixteen
wave64 waves, 28 PCs per wave (448 instruction starts), 2,048 read lanes,
1,024 write lanes, 16 stores, native-private signal `1 -> 0`, and output CRC-32C
`3210199849`. Only C is required for D2H (`required_d2h_mask=4`); the client
independently verifies bit-exact float32 `C=A+B`. Session completion requires
durable type 20, C D2H, UNMAP, owner cleanup, and no CPU/NVIDIA fallback.

This is one submit per owner/session. A second kernel on the same connection,
arbitrary OpenCL images, events/asynchrony, general launch shapes, normal
Triton Python, model operators, multi-token inference, TP, and CCL remain
unaccepted. The CP26 `gpuReadWrite` profile and its three-buffer D2H/quarantine
rules remain unchanged.

## Triton boundary and non-claims

The CP-0021 retained Triton tutorial and HSACO identities are recorded in the
authority JSON. The codec can represent its 48-byte kernarg and 24832-by-256
work-item/workgroup geometry, but the 12-DWORD (48-byte) descriptor preload is
rejected until preload-aware mapping and entry semantics exist. CP25 continues
to return NOT_SUPPORTED instead of stripping or reinterpreting that field.

CP-0027 does not promote the retained 5,408-byte Triton image with 12-DWORD
preload, and it does not exercise a normal launcher, compiler/JIT transport,
performance path, arbitrary gfx950 HSACO, Triton end-to-end, CPU/NVIDIA
fallback, or Qwen inference. The next product gate is the normal Triton Python
vecadd driver/runtime path. Profiling remains conditional on a measured real
operator, layer, or model bottleneck.
