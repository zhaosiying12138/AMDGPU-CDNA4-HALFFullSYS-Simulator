# Host transport v1 pinned dispatch extension

CP-0008 adds `PINNED_DISPATCH_V1` (capability bit 4, byte zero, mask
`0x10`). It is a deliberately narrow execution contract. The machine-readable
authority is [`protocol/host-transport-v1-dispatch.json`](../protocol/host-transport-v1-dispatch.json).
The CP-0004 80-byte envelope, big-endian framing, CRC-32C, identity tuple, and
HELLO-seeded request-ID namespace remain unchanged. Dispatch records carry no
ancillary descriptors.

This extension is not a general GPU API. It permits exactly one protocol-owned
gfx950 fixture: wave64, one compute unit, one workgroup, one wave, and 64 bytes
of input/output. The client chooses only generation-safe handles and the
published fixture digest. It never supplies a pointer, AQL packet, kernarg,
HSACO, code object, FD, geometry, or arbitrary operation.

## Negotiation and records

The capability is selected only when the client both offers and requires bit 4.
CP-0005 queue control, CP-0006 simulated memory, and CP-0007 signal/event must
also be selected. Message types are stable and reserved as follows:

| Type | Name | Payload | Frame |
| ---: | --- | ---: | ---: |
| 11 | `DISPATCH_REQUEST` | 128 bytes | 208 bytes |
| 12 | `DISPATCH_ACK` | 160 bytes | 240 bytes |
| 13 | `DISPATCH_COMPLETION` | 160 bytes | 240 bytes |

All integers in these payloads are big-endian. The base CRC covers the complete
frame with header bytes 64 through 67 zeroed while calculating it. A request is
one `SUBMIT_PINNED` operation. Its payload is laid out without padding:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 2 | dispatch major (`1`) |
| 2 | 2 | dispatch minor (`0`) |
| 4 | 2 | opcode (`SUBMIT_PINNED = 1`) |
| 6 | 2 | flags (`0`) |
| 8 | 8 | queue ID |
| 16 | 8 | queue generation |
| 24 | 8 | shared CP-0005 queue sequence |
| 32 | 8 | fixture ID (`1`) |
| 40 | 8 | input allocation ID |
| 48 | 8 | input generation |
| 56 | 8 | output allocation ID |
| 64 | 8 | output generation |
| 72 | 8 | CP-0007 signal ID |
| 80 | 8 | signal generation |
| 88 | 8 | expected final signed signal bits (`0`) |
| 96 | 32 | exact fixture manifest SHA-256 |

The result payload echoes all queue, allocation, signal, fixture, and sequence
identities, then carries a compact execution summary:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 2 | dispatch major (`1`) |
| 2 | 2 | dispatch minor (`0`) |
| 4 | 4 | base status |
| 8 | 2 | opcode |
| 10 | 2 | reserved zero |
| 12 | 4 | reserved zero |
| 16..95 | 80 | echoed queue/fixture/allocation/signal identities |
| 96 | 8 | trace ID |
| 104 | 8 | input GPU VA |
| 112 | 8 | output GPU VA |
| 120 | 4 | materialized AQL packet CRC-32C |
| 124 | 4 | output CRC-32C (zero in an ACK) |
| 128 | 8 | admission tick |
| 136 | 8 | GPU start tick (zero in an ACK) |
| 144 | 8 | final-store/end tick (zero in an ACK) |
| 152 | 8 | packet-retire tick (zero in an ACK) |

`request_id` in the base header is the sole correlation key. The ACK and
completion both echo the dispatch request ID; `trace_id` is a daemon-lifetime
nonzero ID shared by both records and every retained trace event. A completion
using the active request ID before its ACK, or a foreign/duplicate completion,
poisons the session.

## The pinned fixture

Fixture ID 1 is fully specified in the JSON authority. Its geometry is
`grid=(64,1,1)`, `workgroup=(64,1,1)`, one workgroup and one wavefront of 64
lanes on CU 0. The two live CP-0006 allocations are distinct, each at offset
zero. The live CP-0007 signal is signed one and already has exactly one
unsatisfied `EQ 0` wait admitted by the same client. Input is the exact bytes
`00..3f`, output starts as 64 zero bytes, and the only operation is:

```text
output[lane] = input[lane] XOR 0x5a,  lane = 0..63
```

The expected output is non-identity and has CRC-32C `0x796671ec`. The fixture
contains an exact 64-byte little-endian `AMDKernelCode` descriptor followed by
44 bytes of gfx950 ISA (`code_image_sha256_hex` is
`711b101e0e78c4ac7eae865b5f34779d63f2cc9cba8e0b3025bca99fa646ad49`). The
canonical zero-address 64-byte AQL packet has SHA-256
`e8d8451e27b883a068918315c73c807d59f37f770de987e18689e201ecefb92f`. The
16-byte zero-pointer kernarg template and the five-component manifest are also
hashed and frozen. The request carries the full 32-byte manifest hash
`7500741873f9d39848e57f0aa9ffc6454df7db87b93e1c046501f54db1b7543c`; a
different digest is a canonical `MALFORMED` request.

The daemon materializes only the code-object and kernarg AQL addresses. The
AQL `completion_signal` field remains the HSA special handle zero and is never
the CP-0007 handle. At real `finishPkt` retirement, a generation-checked
listener mirrors packet completion by storing signed zero to the CP-0007 signal.
If packet retirement is tick `R`, the signal store is observed at `R` and its
CP-0007 wait completion is exactly `R+1` without overflow. That canonical signal
completion is ordered before the dispatch completion. Only after the runtime
has validated both records may it perform D2H and accept the 64 exact golden
bytes.

## Admission versus execution

`DISPATCH_ACK/OK` is an admission record, not a completion record. It means the
queue sequence was committed, all four live generation-safe handles were pinned,
the fixture preconditions were checked, and internal AQL/kernarg/code storage
was materialized. The daemon rings the real HSA doorbell exactly once only
after the complete successful ACK send. It says
nothing about packet fetch, GPU issue, CU stores, signal mutation, retirement,
or output bytes. ACK success has a nonzero trace ID and VAs and a packet CRC,
but output CRC and start/end/retire ticks are zero.

`DISPATCH_COMPLETION/OK` is emitted only after the established gem5 path has
performed HSAPacketProcessor fetch/process, GPUCommandProcessor submission,
GPUDispatcher/CU execution, all 64 global stores, packet retirement, and the
CP-0007 signal mirror. It repeats the ACK summary, supplies the golden output
CRC, and satisfies `admission < start <= end <= retire` with retirement more
than one tick after admission. A non-OK completion is noncanonical: execution
failure is retained in trace and closes the session rather than inventing a
synthetic result.

The retained JSONL trace must contain, in order, admission, AQL registration and
publication, HSAPP fetch, GPU command processor submission, GPU dispatcher and
CU start, global-store completion, dispatcher completion, packet retirement,
CP-0007 signal mirror, and wire completion. Every event carries request/trace
correlation and the full code/AQL hashes. The authority's `event_fields` table
fixes every event-specific JSON key, type, constant, and cross-record identity;
emitters and validators may not substitute aliases. Trace summary disagreement, missing
CU provenance, or a matching output produced by host arithmetic fails CP-0008.

Issued-work disconnect evidence uses the same common trace fields but is kept
outside the successful twelve-event sequence. `dispatch_owner_disconnected`
records the exact admitted/issued/task-known/retired phase, internal queue and
packet identity, whether quarantine is required, and which completions had
already been emitted. Work disconnected before retirement must later emit one
`abandoned_packet_retired` only after real `finishPkt` correlation and
quarantine release; it records that neither the CP-0007 mirror nor a wire
completion was emitted. An admitted but unissued disconnect rolls back without
that retirement event, while a retired pre-wire disconnect suppresses the
remaining outbound records without inventing a second retirement.

## Validation and failure precedence

The daemon evaluates the published precedence list mechanically: malformed
envelope/CRC and unsafe ancillary data drop and close; correlatable canonical
field errors receive `MALFORMED`; version, identity, topology, capability, and
shared request-ID checks follow; stale generations, queue sequence, busy state,
fixture byte preconditions, resources, and materialization errors are then
reported in order. A failed ACK echoes structurally decoded identities but
zeros all result values and never creates a completion. An admitted dispatch
consumes its request ID and queue sequence even if a later client wait times
out.

Submit forms one absolute `CLOCK_MONOTONIC` deadline and ends after a canonical
ACK publishes its admission ticket, including the dispatch request ID used by
the retained trace. Each later wait call forms a new absolute
deadline. Before a complete ACK, timeout, cancellation, EOF, or protocol failure
after a complete request send is indeterminate and poisons/closes the session.
After a canonical ACK, wait timeout or cancellation is retryable without
resending the request or allocating a request ID; the pending completion remains
buffered. EOF, daemon execution failure, completion send failure, foreign
records, and trace inconsistency poison. Disconnect invalidates client handles
and callbacks; already-issued GPU storage is quarantined until retirement, with
no signal or completion emitted to the disconnected owner.

## Acceptance boundary

CP-0008 proves one source-pinned, traceable gfx950 AQL execution and its
generation-safe queue/memory/signal wiring. It does not prove generic HSACO or
ELF/MsgPack loading, ROCr/libhsakmt (P2), HIP, OpenCL, Triton, PyTorch, vLLM,
multi-CU or multi-workgroup scheduling, collectives, tensor parallelism, or
performance timing. Bridge arithmetic, CPU fallback, direct dispatcher calls
that bypass HSAPP, and one-tick timers are explicitly outside the boundary.

## Golden records

The JSON authority contains byte-complete request, admission ACK, and execution
completion goldens. They are respectively 208, 240, and 240 bytes with CRCs
`cf679df8` (request), `5c3070d8` (ACK), and `5cf2625b` (completion). Independent
tests rebuild the headers and payloads,
recompute CRC-32C, verify all offsets and hashes, and assert that ACK and
completion correlation is exact.
