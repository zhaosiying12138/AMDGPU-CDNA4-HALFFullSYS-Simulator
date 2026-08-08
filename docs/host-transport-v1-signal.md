# Host transport v1 signal and event extension

`SIGNAL_EVENT_V1` is the CP-0007 bridge-private signal boundary. It adds a
bounded signed 64-bit signal lifecycle and exactly-once asynchronous wait
completion without changing the frozen host-transport envelope or the queue and
memory payloads. It does not create a KFD event object, expose signal storage to
GPU packets, or claim dispatch or kernel completion.

The machine-readable authority is
[`protocol/host-transport-v1-signal.json`](../protocol/host-transport-v1-signal.json).
This document explains the state and failure boundaries that accompany its byte
layout.

## Negotiation and framing

`SIGNAL_EVENT_V1` is capability bit 3 (byte 0, mask `0x08`). A conforming client
offers and requires it together; offered-only is not selected. Selection is
independent of `QUEUE_CONTROL_V1` and `SIMULATED_MEMORY_V1`.

The extension uses the unchanged 80-byte big-endian base header and a fixed
64-byte payload. `SIGNAL_REQUEST`, `SIGNAL_ACK`, and `SIGNAL_COMPLETION` have
message types 8, 9, and 10. Every frame is therefore exactly 144 bytes and its
CRC-32C covers the complete frame with the header checksum field cleared. Signal
records never carry ancillary descriptors. Any descriptor, truncated ancillary
data, or malformed control message is unsafe framing and closes the connection.

The request payload contains protocol version, opcode, flags, signal ID,
generation, wait sequence, signed value bits, condition, and two reserved
words. The result payload contains protocol version, status, opcode, signal ID,
generation, wait sequence, signed value bits, a `ready` word, and simulation
tick. All flags and reserved fields are zero.

Signal values use exact two's-complement 64-bit representations on the wire.
They are compared as signed `int64` values. The conditions intentionally match
the HSA numeric ABI: `EQ=0`, `NE=1`, `LT=2`, and `GTE=3`.

## Lifecycle and value operations

The daemon owns 1024 fixed slots. `CREATE` selects the lowest free slot, assigns
a daemon-lifetime monotonically increasing nonzero generation, stores the
arbitrary signed initial value, and returns the slot and generation. Slots may
be reused only with a different generation. Neither an ID nor a generation may
wrap. Exhausted slots or generation space return `RESOURCE_EXHAUSTED`.

`LOAD` returns the current value. `STORE` atomically replaces it and returns the
stored value. `DESTROY` releases a live signal only when no wait or completion
callback remains pending. A stale, foreign, freed, out-of-range, or incorrectly
generated handle returns `PROTOCOL_STATE`; a live signal with a pending wait
returns `BUSY`.

The `sim_tick` in a successful `CREATE`, `LOAD`, or `DESTROY` ACK is the
daemon's admission tick and is intentionally opaque: it may be any `u64` value,
including `UINT64_MAX`, and the runtime does not compare it with a predicted
client-side tick. A `STORE` ACK likewise carries its daemon admission tick. If
that store is the first store to satisfy an armed wait, its tick `S` must be
below `UINT64_MAX` so the owed completion at `S+1` is representable; a store
that satisfies no armed wait may use any `u64` tick. A successful `WAIT` ACK
has admission tick `T`, which must be below `UINT64_MAX` for the same reason.
The runtime enforces only these bounds and the completion relationships for
ticks it has actually received: an immediate completion is exactly `T+1`, and
an armed completion is exactly `S+1` and greater than `T`.

Ownership includes the client file descriptor generation in addition to the
wire signal ID and signal generation. Disconnect cancels every wait and callback
and releases every signal belonging to that owner. Runtime signal handles have
unique ownership; copied aliases are invalid after destroy or instance close.

## Wait admission and completion

`WAIT` carries a per-signal sequence, signed compare value, and condition. An
accepted sequence is nonzero and exactly the previous accepted sequence plus
one. A rejected request does not advance it. At most one armed-or-scheduled wait
exists per signal and at most eight exist in the session.

At admission tick `T`, gem5 samples the signal value and evaluates the signed
predicate:

- If it is true, the ACK returns the snapshot with `ready=1`; that snapshot is
  captured for a completion due at `T+1`.
- If it is false, the ACK returns the snapshot with `ready=0`; the wait remains
  armed.
- The first later `STORE` whose committed value satisfies an armed wait captures
  that value at store tick `S` and makes the completion due at `S+1`.

The completion is exactly one `SIGNAL_COMPLETION` with status `OK`, opcode
`WAIT`, `ready=0`, and the originating request ID, signal ID, generation, and
sequence. Its value is the captured satisfying snapshot. Once a wait becomes
ready it is never re-evaluated, so a later store cannot alter the event. There
is no failed completion form: lifecycle and resource failures are terminal
ACKs, and any non-OK completion is a protocol error. A ready wait remains
counted, and continues to make `DESTROY` busy, while its completion is queued or
blocked by socket backpressure. Its resource is released only after the complete
frame is successfully sent. Send failure or outbound overflow disconnects the
owner and performs the normal generation-safe cleanup.

Tick overflow is atomic. A `WAIT` at `UINT64_MAX` returns `INTERNAL` without
advancing its sequence or creating a wait. At that tick, a `STORE` that would
first satisfy an armed wait returns `INTERNAL` and changes neither value nor
waiter; a store that satisfies no waiter may still commit. Tick
`UINT64_MAX-1` may schedule a completion at `UINT64_MAX`.

## Event-queue and stream ordering

Host `PollEvent` callbacks only schedule service. Decode, request-ID admission,
lifecycle mutation, value access, predicate evaluation, ACK construction, and
completion construction all run on the gem5 event queue. Record service gives
all signal operations a deterministic total order. A `STORE` commits before its
ACK is constructed and before newly satisfied waits are queued.

The outbound bound is 25 records: up to eight queued-but-unsent prior queue
completions, eight newly admitted queue completions, eight signal waits or
completions, and one synchronous ACK. Both queue cohorts are required because
constructing a queue completion releases its inflight command before the socket
send succeeds. A callback revalidates the owner file descriptor generation,
signal ID and generation, wait sequence, and originating request ID before it
emits an event.

The HELLO-seeded request-ID tracker is shared across queue, memory, and signal
records. It is strictly increasing and never wraps. Once structure, extension
version, identity, and capability are valid, the request consumes its ID even
when lifecycle, busy, resource, or tick validation rejects it.

While any established ACK or completion is awaited, the runtime may receive a
canonical completion for previously acknowledged queue or signal work. It
validates and bounded-buffers that completion. A completion carrying the active
request ID before its ACK, an unknown record, a duplicate, a foreign identity,
or a noncanonical result poisons the session. Existing queue completion rules
remain unchanged.

## Deadlines, cancellation, and poison

Signal operations use one `CLOCK_MONOTONIC` absolute deadline formed after
synchronous argument validation. An expired deadline wins over simultaneous
cancellation. The runtime checks both before consuming a buffered completion.

After a complete request record is sent, timeout, cancellation, EOF, or a
noncanonical response before the canonical `SIGNAL_ACK` makes the remote result
indeterminate, so the runtime poisons and closes the connection. A canonical
non-OK ACK is determinate and leaves the session reusable.

After a successful `WAIT` ACK, timeout or cancellation while waiting for its
completion is different: the accepted wait remains pending, any already
buffered completion remains unconsumed, and the session remains usable. A retry
with the same signal, condition, and compare value waits on that existing event
without allocating a request ID or sending another `WAIT`. A different
predicate, or destroy, returns local `BUSY`; load and store remain available, so
a store can satisfy the wait. EOF or protocol failure during the completion
wait still poisons the session. Result output is published only after the full
completion validates.

## Error order and scope

Unsafe framing is dropped and closed first. Correlatable payload shape errors
are `MALFORMED`, followed by extension version, daemon identity, connection,
epoch, capability, shared request-ID, handle/sequence, busy, resource, and
internal tick/scheduling checks in that order. Every canonical failed ACK echoes
the decoded opcode, signal ID, generation, and sequence, uses the actual daemon
header identity, and zeros value, ready, and simulation tick. It never produces
a completion.

This checkpoint deliberately excludes separate KFD event handles, signal
arithmetic and compare-and-swap, GPU-visible signal memory, queue or packet
completion linkage, packet submission, code objects, kernel execution, and
collectives. `SIGNAL_COMPLETION` is the only event primitive introduced here.
