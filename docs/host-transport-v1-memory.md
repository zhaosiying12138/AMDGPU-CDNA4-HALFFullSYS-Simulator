# Host Transport v1 Simulated Memory

CP-0006 adds a bounded simulated-memory data path to the accepted host
transport. It performs real cross-process byte transfer into storage owned by
the gem5 process. It deliberately does not claim packet-visible GPU VRAM,
SDMA timing, packet submission, code-object loading, kernel execution, or
collectives.

The mechanical contract is
`protocol/host-transport-v1-memory.json`. The accepted 80-byte envelope,
big-endian encoding, CRC-32C, identity fields, and request-ID rules remain
unchanged. Existing handshake and queue golden frames remain byte-identical.

## Negotiation And Frames

`SIMULATED_MEMORY_V1` is capability bit 2 (byte zero, mask `0x04`). A memory
client must both offer and require it, and the daemon selects it if and only if
both bits are present. Offered-only is not selected; required-not-offered is
the base subset error. The capability is independent of queue control. Message
type 6 is `MEMORY_REQUEST` and type 7 is `MEMORY_ACK`; both
are fixed 144-byte records with a 64-byte payload. CP-0006 memory operations
have one synchronous terminal ACK and no asynchronous completion.

The request payload carries memory version 1.0, opcode, allocation ID,
generation, offset, byte count, one opcode argument, and two zero reserved
words. `ALLOC` requests a nonzero size and alignment of 4 KiB or 64 KiB.
`FREE` names an allocation. `COPY_H2D` carries a zero-extended content CRC-32C
and `COPY_D2H` carries a zero argument. Zero-size operations and overflowing
ranges are rejected.

A successful allocation ACK returns a slot ID from 1 through 1024, a nonzero
generation, the exact slot VA, requested size, and alignment. Copy ACKs return
the exact offset, count, and verified content CRC. Every ACK is one 144-byte
type-7 record without ancillary data, has version 1.0, echoes the request ID,
uses the actual daemon UUID/current connection/current epoch, and has canonical
reserved fields. A failed ACK uses a known nonzero status, echoes the decoded
opcode, allocation ID, and generation, and has zero values and tick. Any ACK
shape, correlation, identity, value, descriptor, seal, size, or CRC mismatch
poisons the runtime session.

## Carrier Contract

Control and data remain separate. `ALLOC` and `FREE` carry no descriptor.
Each H2D or D2H request carries exactly one anonymous, sealable regular staging
file through `SCM_RIGHTS`; ACKs never carry a descriptor. The properties the
daemon can verify are an anonymous seal-capable regular object, not its syscall
provenance. A conforming runtime creates
that file with `memfd_create(MFD_CLOEXEC | MFD_ALLOW_SEALING)`, sets mode 0600,
and gives it the exact transfer length. The receiver requires a regular file,
zero link count, matching owner, exact size and mode, receiver-side
`FD_CLOEXEC` established by `MSG_CMSG_CLOEXEC`, supported seals, and the
opcode-specific access mode. Receiver `FD_CLOEXEC` does not prove the sender's
creation flags. File-position state is ignored and all data I/O starts at
offset zero through `pread` or `pwrite`.

For H2D, the exact seals are `F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE |
F_SEAL_SEAL`. gem5 reads the whole staging object into scratch, verifies the
CRC, prepares every sparse destination chunk, and only then commits bytes.
A bad CRC is a determinate `MALFORMED` result and changes no simulated byte.

For D2H, the initial access mode is `O_RDWR` and the exact initial seals are
`F_SEAL_SHRINK | F_SEAL_GROW`. gem5 snapshots the requested simulated bytes,
writes the complete staging object, adds `F_SEAL_WRITE | F_SEAL_SEAL`, verifies
the final four seals, re-reads the now-immutable complete contents, computes
their CRC, and only then sends success. A client writable mapping that blocks
the write seal is `UNAUTHORIZED`; staging-space exhaustion is
`RESOURCE_EXHAUSTED`; other I/O, sealing, or verification failures are
`INTERNAL`. The runtime validates the ACK, size, final seals, and CRC into
private scratch before changing the caller's destination buffer.

Truncated, malformed, unknown, or multi-descriptor control data is closed and
the connection is dropped. A complete memory request with the wrong expected
descriptor count receives `MALFORMED`. A correctly cardinalized but unsafe
carrier receives `UNAUTHORIZED`. Any descriptor attached to a handshake,
queue record, ACK, or unknown message retains the earlier drop-and-close rule.
Every received descriptor enters move-only RAII immediately.

## Allocation State

The bridge owns 1024 reusable slots. A single allocation is at most 2 GiB,
total live requested bytes are at most 4 GiB, and a transfer is at most 16 MiB.
For slot ID `n`, the VA is exactly `0x0000100000000000 + (n - 1) * 2 GiB`,
using checked arithmetic. Each slot has a daemon-lifetime monotonic nonzero
generation. Reuse changes the generation; generation exhaustion never wraps.

Backing is sparse in 64-KiB chunks. Missing chunks read as zero, so allocation
does not eagerly consume the requested host RAM. H2D prepares all affected
chunks before swapping them into the allocation. Disconnect frees every slot
owned by the client file-descriptor generation and closes pending carriers.
The returned address is a bridge-owned functional simulated VA, not yet an
address accepted by AMDGPU packet execution.

Poll callbacks only schedule gem5 events. Protocol decode, handle/range
validation, carrier I/O, sparse-state mutation, and ACK construction all run
from the scheduled event-queue service. This gate provides functional byte
ownership and transfer, not device-memory timing fidelity.

## Failure And Retry

The session-wide request-ID sequence is shared with queue control, starts from
HELLO, increases strictly, and never wraps. Structural, version, identity, and
capability checks precede request-ID admission. Every admitted request that can
receive a canonical ACK consumes its ID, including lifecycle, resource,
carrier-policy, carrier-I/O, and H2D CRC failures.

After a complete request and required descriptor are sent, timeout,
cancellation, EOF, or a noncanonical ACK leaves remote mutation uncertain.
The runtime poisons and closes the session; it does not replay the request.
A canonical non-OK ACK is determinate and leaves the session usable. Failures
before a record is sent do not poison the session.

One `CLOCK_MONOTONIC` absolute deadline is formed after synchronous argument
validation and covers memfd creation, staging, sealing, send, ACK, and
successful D2H post-ACK validation. Deadline precedes cancellation. After send
but before a canonical ACK, either condition poisons the session. ALLOC, FREE,
H2D, and every non-OK result are terminal immediately after canonical ACK
validation. Only successful D2H checks deadline and cancellation again before
the final caller-buffer copy; either condition then leaves output unchanged and
the known session usable, so the caller may issue a new D2H. No late check turns
an already completed operation into cancellation.

While waiting for `MEMORY_ACK`, the runtime may receive a canonical CP-0005
`QUEUE_COMPLETION` for a previously acknowledged queue command. It validates
and buffers that completion without changing the active memory exchange. A
completion carrying the active memory request ID, any noncanonical completion,
or any other record type poisons and closes the session. Instance close
invalidates all local allocation handles; copied C pointer aliases are outside
the API contract after free or close.

## Acceptance Boundary

Acceptance requires byte-exact control goldens, carrier seal and descriptor
tests, sparse zero-fill, cross-chunk roundtrip, slot reuse with generation
change, range and resource rejection, disconnect cleanup, FD leak checks,
CRC failure without partial mutation, ACK ambiguity poisoning, queue
completion interleaving, and a real runtime-to-gem5 allocation/copy/free gate.
The evidence must not describe these bytes as packet-visible VRAM or GPU
execution.
