# Host Transport v1 Queue Control

CP-0005 adds a deliberately small extension to the accepted host-transport-v1
handshake. The extension is a versioned, bounded control queue. It is not a
GPU queue and it does not expose a virtual address, a packet, a code object, a
kernel, or a memory operation.

The base 80-byte envelope, big-endian encoding, identity fields, request-ID
rules, CRC-32C, deadlines, and established-session state rules remain those
of `protocol/host-transport-v1.json`. The extension is described mechanically
in `protocol/host-transport-v1-queue.json`.

## Negotiation

`QUEUE_CONTROL_V1` is capability bit 1 (byte zero, mask `0x02`). A queue
client must both offer and require this bit. A daemon selects it only when it
was offered and required. Existing clients that do not mention the bit remain
valid CP4 clients and never create queue state; a queue request on such a
session is rejected as `UNSUPPORTED_CAPABILITY`.

## Frames

Message type 3 is `QUEUE_REQUEST`, type 4 is `QUEUE_ACK`, and type 5 is
`QUEUE_COMPLETION`. Each has exactly 64 payload bytes. All multi-byte fields
are big-endian. Request headers must match the negotiated daemon UUID,
connection ID, and job epoch, and every request ID is nonzero and unique for
the session. Queue request IDs increase strictly after the HELLO request ID and
never wrap; exhaustion fails without sending or mutating queue state. Unknown
flags, reserved bytes, versions, opcodes, lengths, CRCs, or identity fields
fail closed.

Error precedence is deterministic. An invalid envelope, CRC, message type,
frame or payload length, header reserved field, or uncorrelatable request is
silently dropped and the connection closes. In a complete 64-byte payload, a
correlatable queue-flag, payload-reserved, opcode, or opcode-canonical error
receives `MALFORMED`.
Next come unsupported queue version, daemon UUID mismatch, connection-state
mismatch, epoch mismatch, missing queue capability, non-increasing request ID,
then the queue lifecycle result. The request-ID tracker advances only after
structural, version, identity, and capability validation succeeds.

The request payload contains version, opcode, flags, queue handle, generation,
sequence, two opcode arguments, and two reserved words. `CREATE` uses `arg0`
as a depth from 1 through 64. `DESTROY` identifies a client-owned handle and
can only complete when no command is pending. `DOORBELL` accepts only command
kinds `0` (`NOOP`), `1` (`CONTROL_TEST`), and `2`
(`CONTROL_ERROR_TEST`), and its sequence must increase by exactly one. Kind 2
is a control-only error fixture: it is accepted, then completes one tick later
with `INTERNAL`, `value=2`, and `error_code=1`. The result payload echoes the
operation and handle and carries a wire status, value, error code, and
simulated tick.

Unused request fields are canonical zero. `CREATE` requires zero handle,
generation, sequence, and `arg1`; `DESTROY` requires zero sequence and
arguments; `DOORBELL` requires nonzero handle, generation, and sequence with
zero `arg1`. A successful CREATE ACK returns the accepted depth. A successful
DESTROY ACK has zero value and error code. A successful DOORBELL ACK has zero
value and error code; its completion carries the command kind in `value`.
Failure ACKs echo the structurally decoded opcode, handle, generation, and
sequence and otherwise keep value and error code zero. They echo the request
ID but never reflect an invalid request identity: the response header uses the
actual daemon UUID, current connection ID, and actual epoch.

There are at most eight queues, depth 64 per queue, and eight in-flight
commands per daemon. A successful doorbell is acknowledged immediately and
produces one completion event one simulated tick later. Both the ACK and the
completion echo the originating nonzero request ID. Completion callbacks carry
that request ID, the client file-descriptor generation, and the queue
generation; stale callbacks are discarded after disconnect or descriptor
reuse. Queue state is mutated only by the gem5 event queue.

After a complete queue request record is sent, losing or failing to validate
its ACK leaves the remote result unknown. The runtime must poison and close the
session so daemon disconnect cleanup removes all queue state; it must not reuse
that session. A canonical non-OK ACK is a determinate rejection and does not
poison the session. Timeout or cancellation while waiting for the completion
of an already acknowledged doorbell leaves the sequence pending and `wait` may
be retried, including when the completion is already buffered. Deadline is
checked before cancellation and both are checked before consuming that buffer.
A malformed, foreign, or noncanonical completion poisons the session. The only
canonical non-OK completion in CP-0005 is the kind-2 control error fixture.

## Acceptance boundary

The CP-0005 gate covers codec vectors, malformed and stale-handle rejection,
bounded queue behavior, ordered doorbells, disconnect cleanup, and the
runtime-to-gem5 completion/error path. It does not claim allocation, copying,
packet submission, code-object loading, kernel execution, collectives, or
PyTorch/vLLM execution. Those require later checkpoints and separate source
and ABI evidence.

The runtime C `sagr_queue_t` is a unique-ownership opaque handle. Callers must
not copy it for later reuse. Successful queue destruction clears the supplied
handle, and instance close invalidates every queue handle owned by that
instance; passing a copied alias after either operation is outside the C API
contract. Deterministic stale-handle rejection in this protocol refers to the
wire `(queue_id, generation)` pair while the local opaque handle is alive.
