# GemSim Host Transport Framing and Handshake 1.0

This document is the normative protocol contract between
`self-amdgpu-runtime` and a gem5 host daemon. The machine-readable companion is
`protocol/host-transport-v1.json`. An implementation conforms only when its
wire output matches both documents and the golden frames byte for byte.

The handshake establishes a control connection and transport identity. It
does not claim GPU execution, memory transfer, descriptor passing, shared
memory, doorbells, request cancellation, transparent reconnect, collectives,
or any HIP/HSA/OpenCL operation.

Normative terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their usual
requirements-language meanings.

## 1. Transport and record boundary

The endpoint MUST be an absolute Linux pathname. The socket MUST use
`AF_UNIX`, `SOCK_SEQPACKET`, and protocol zero. One seqpacket record is exactly
one protocol frame. Implementations MUST NOT coalesce frames or split one frame
across records. All integers are unsigned and encoded big-endian. Byte arrays
are copied verbatim. Wire data MUST be serialized field by field and MUST NOT
be produced from an in-memory C or C++ structure layout.

The absolute maximum record is 65,536 bytes. A HELLO or HELLO_ACK record is
additionally limited to 4,096 bytes. A receiver MUST detect `MSG_TRUNC` and
`MSG_CTRUNC`; either condition invalidates the record and connection.

## 2. Common 80-byte header

| Offset | Bytes | Field | Requirement |
| ---: | ---: | --- | --- |
| 0 | 8 | magic | Exact bytes `47 53 49 4d 52 50 43 00` (`GSIMRPC\0`) |
| 8 | 2 | framing major | `1` |
| 10 | 2 | framing minor | `0` |
| 12 | 2 | header bytes | `80` |
| 14 | 2 | message type | HELLO `1`, HELLO_ACK `2` |
| 16 | 4 | flags | `0` |
| 20 | 4 | payload bytes | Exact record size minus 80 |
| 24 | 8 | request ID | Nonzero; ACK echoes HELLO |
| 32 | 16 | daemon instance UUID | Expected identity or zero wildcard in HELLO; actual identity in ACK |
| 48 | 8 | connection ID | Zero in HELLO; nonzero in successful ACK |
| 56 | 8 | job epoch | Expected epoch or zero wildcard in HELLO; actual epoch in ACK |
| 64 | 4 | CRC32C | Section 3 |
| 68 | 4 | reserved0 | Zero |
| 72 | 8 | reserved1 | Zero |

A daemon instance UUID identifies one daemon process lifetime, MUST be nonzero,
and MUST NOT be reused by a later process lifetime. It is separate from the job
UUID. The daemon's actual job epoch MUST be nonzero. A connection ID identifies
one accepted connection, MUST be newly generated and nonzero on success, and
MUST NOT be reused during that daemon process lifetime.

For HELLO, connection ID MUST be zero. For successful HELLO_ACK, status is OK,
daemon instance UUID and connection ID MUST be nonzero, and all returned
identity fields MUST describe the accepting daemon. A client validates the
request ID and nonce echo before interpreting any ACK status. It validates the
expected instance, epoch, topology, selected capabilities, and connection ID
only for an OK ACK; an identity-rejection ACK necessarily reports an identity
that differs from the client's expectation.

An invalid magic, record/header/payload size, CRC, reserved field, request ID,
message type, or HELLO session field invalidates the envelope. The daemon MUST
drop the connection without an ACK. Invalid peer credentials follow the same
drop-without-ACK rule.

On a newly accepted connection the daemon accepts only HELLO. Receiving
HELLO_ACK or any unknown message type in that state is silently dropped, which
also prevents two misconfigured servers from forming an ACK loop. A client
waiting for HELLO_ACK treats any other message type as a protocol error. After
an OK ACK, a second HELLO on the same connection receives the canonical failed
ACK with PROTOCOL_STATE and the connection is closed. Envelope and structural
HELLO validation still precede this state check; after a structurally valid
HELLO is decoded, PROTOCOL_STATE precedes version, capability, identity, and
topology negotiation.

## 3. CRC32C

The checksum is CRC-32C Castagnoli using the reflected polynomial
`0x82f63b78`, initial value `0xffffffff`, and final XOR `0xffffffff`. It covers
the complete header and payload with header bytes 64 through 67 treated as
zero. The mandatory independent check vector is:

```text
CRC32C("123456789") = 0xe3069283
```

The receiver MUST validate record bounds before calculating the checksum. It
first proves that the record is between 80 bytes and the applicable maximum,
then compares the 32-bit payload length to `record_size - 80`. It MUST NOT add
the untrusted payload length to 80 when validating a received record.

## 4. HELLO payload

HELLO has message type `1` and a 96-byte fixed prefix followed by TLVs.

| Offset | Bytes | Field | Requirement |
| ---: | ---: | --- | --- |
| 0 | 2 | minimum major | `1` in version 1.0 |
| 2 | 2 | minimum minor | `0` |
| 4 | 2 | maximum major | `1` |
| 6 | 2 | maximum minor | `0` |
| 8 | 16 | client nonce | Nonzero |
| 24 | 32 | offered capabilities | Section 7 |
| 56 | 32 | required capabilities | Section 7 |
| 88 | 4 | receive maximum record | Inclusive range 4,096 through 65,536 |
| 92 | 2 | role | RUNTIME `1` |
| 94 | 2 | reserved | Zero |

The version range MUST be lexicographically ordered; an inverted range is
MALFORMED. The CP-0004 daemon supports only version 1.0 and selects it whenever
`minimum <= (1, 0) <= maximum`; an ordered range without 1.0 is
UNSUPPORTED_VERSION. The default runtime sends the exact range 1.0 through 1.0.
CP-0004 requires every successful session to both offer and require capability
bit zero. A HELLO that offers bit zero MUST contain exactly one critical topology
identity TLV. A HELLO that does not offer bit zero MUST NOT contain that TLV
and is rejected as UNSUPPORTED_CAPABILITY because topology identity is
mandatory for a CP-0004 session.

## 5. HELLO_ACK payload

HELLO_ACK has message type `2` and an 80-byte fixed prefix followed by TLVs.

| Offset | Bytes | Field | Requirement |
| ---: | ---: | --- | --- |
| 0 | 2 | selected major | `1` on success, zero on failure |
| 2 | 2 | selected minor | `0` |
| 4 | 4 | status | Section 8 |
| 8 | 16 | client nonce echo | Exact HELLO nonce |
| 24 | 16 | server nonce | Nonzero on success, zero on failure |
| 40 | 32 | selected capabilities | Negotiated subset on success, all zero on failure |
| 72 | 4 | maximum record | Negotiated limit on success; daemon configured limit on failure |
| 76 | 2 | role | DAEMON `2` |
| 78 | 2 | reserved | Zero |

The ACK MUST echo the HELLO request ID and client nonce. A successful ACK MUST
select the highest common version and every required capability. The CP-0004
daemon supports only version 1.0 and `TOPOLOGY_IDENTITY_V1`. Its successful ACK
MUST contain exactly one topology identity TLV describing the daemon.

A HELLO is correlatable only when its envelope is valid, its complete 96-byte
fixed prefix is present, and its client nonce is nonzero. A shorter payload or
zero nonce is silently dropped. Other malformed fields in a correlatable
HELLO, and all negotiation and identity failures, receive a HELLO_ACK carrying
the matching request ID and nonce.

Every failed ACK has one canonical shape: selected version `(0, 0)`, actual
nonzero daemon UUID and actual epoch in the header, zero connection ID, zero
server nonce, all-zero selected capabilities, the daemon's configured maximum
record, and no TLVs. It MUST NOT establish a connection. A client validates
the failed ACK envelope, request ID, nonce echo, canonical failure fields, and
known nonzero status, then returns that peer status without applying success
identity constraints. Because the client does not independently know the
daemon's configured maximum, it validates only that failure `max_record` is in
the legal range; daemon conformance tests verify equality to its configuration.
In `AwaitHello`, error precedence is: malformed HELLO payload/TLV,
unsupported version, unsupported capability, instance mismatch, topology
mismatch, busy, resource exhaustion, then internal error. In any later state,
a structurally valid HELLO receives protocol state before those negotiation
checks; structural malformed remains first. Authorization is checked before
protocol parsing and is therefore silent.
An ACK status outside the defined values in Section 8 is a protocol error and
never establishes a connection.

## 6. TLV encoding

Every TLV starts with `type:u16`, `flags:u16`, and `value_length:u32`, followed
by the value and zero padding to an 8-byte boundary. The only allowed flag is
critical bit `0x0001`; every other flag bit is invalid. A decoder first proves
that at least eight bytes remain, then requires `value_length <= remaining -
8`, computes padding from that already-bounded length, and proves the padding
fits the remaining record. It MUST NOT validate by adding an untrusted length
first. Duplicate types, malformed lengths, nonzero padding, or invalid flags
make the HELLO malformed.

An unknown optional TLV is skipped. A well-formed unknown critical TLV returns
UNSUPPORTED_CAPABILITY. Known TLVs MUST carry their specified flags and exact
value length.

Topology identity has type `1`, critical flags `1`, and value length `24`:

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 16 | job UUID |
| 16 | 4 | rank |
| 20 | 4 | world size |

The server's job UUID MUST be nonzero, world size MUST be at least one, and
rank MUST be less than world size. In HELLO, a zero job UUID combined with rank
`UINT32_MAX` and world size zero is the topology wildcard. Otherwise all three
fields MUST exactly match the daemon. Partial wildcard combinations are
TOPOLOGY_MISMATCH. A successful ACK always returns the exact server topology.

## 7. Capability negotiation

Each bitmap is 32 bytes. Capability bit `n` is byte `n / 8`, mask
`1 << (n % 8)`; this numbering is independent of integer byte order. Bit zero
is `TOPOLOGY_IDENTITY_V1` and authorizes topology TLV type 1.

Required capabilities MUST be a subset of offered capabilities. Selected
capabilities are `offered AND server_supported`. Success requires every
required bit to be selected and, for CP-0004, requires bit zero to appear in
offered, required, and selected capabilities.
The CP-0004 server-supported bitmap contains only bit zero. Topology TLV type 1
MUST appear exactly once if and only if bit zero is offered in HELLO or selected
in a successful ACK. A failed ACK contains neither the bit nor the TLV.
Capabilities MUST NOT be advertised before their complete behavior is
implemented.

## 8. Status values and identity matching

| Value | Name | Meaning |
| ---: | --- | --- |
| 0 | OK | Connection established |
| 1 | MALFORMED | Valid envelope, invalid HELLO payload or TLV |
| 2 | UNSUPPORTED_VERSION | No supported protocol version |
| 3 | UNSUPPORTED_CAPABILITY | Required or critical extension unsupported |
| 4 | INSTANCE_MISMATCH | Non-wildcard daemon UUID differs |
| 5 | TOPOLOGY_MISMATCH | Non-wildcard job/rank/world differs |
| 6 | UNAUTHORIZED | Policy result; peer-credential rejection is silent |
| 7 | BUSY | One active client already owns the daemon |
| 8 | RESOURCE_EXHAUSTED | Bounded host resource unavailable |
| 9 | PROTOCOL_STATE | Message is invalid for connection state |
| 10 | INTERNAL | Daemon failed without a more precise stable status |

HELLO daemon UUID, job UUID, epoch, rank, and world size may be all-wildcard as
defined above or exact. A nonzero instance UUID, nonzero epoch, or non-wildcard
topology is an equality constraint. A daemon MUST NOT silently substitute an
identity. Only one active client is accepted initially; a second valid HELLO
receives BUSY.

A generic diagnostic client MAY use the complete wildcard tuple and inspect the
returned identity. A TP or other multi-instance supervisor MUST instead supply
a nonzero expected job UUID and epoch plus exact rank and world size for every
rank. Endpoint selection alone is not a sufficient TP identity assertion.

## 9. Peer credentials and endpoint lifecycle

Both endpoints MUST use `SO_PEERCRED`, and the daemon and runtime effective
UIDs MUST be equal. PID and GID are diagnostic values, not persistent
identities. Credential authorization occurs immediately after accept and
before reading protocol data. Cross-UID transport requires a future endpoint
DAC and authorization profile and is not part of v1.

The socket parent directory MUST already exist, be owned by the daemon's
effective UID, and have no group or other permission bits. The daemon MUST NOT
traverse a symlink as the endpoint or lock file. It MUST retain an exclusive
`flock` on a same-directory endpoint lock file for the entire listener
lifetime; the lock file itself is retained across runs.

The absolute namespace containing that parent is a deployment trust boundary.
Every ancestor directory entry from `/` through the endpoint parent MUST be
protected from rename, exchange, or mount rebinding by an untrusted UID. A 0700
endpoint parent protects entries inside that directory, but does not protect
the parent's own name in a group- or world-writable non-sticky ancestor. The
parent filesystem MUST support hard links to pathname socket inodes.

If the endpoint exists, the daemon MUST inspect it relative to a trusted parent
directory descriptor and reject symlinks and non-sockets. Before probing it
records the socket's device, inode, owner, and type. While holding the lock, it
performs a bounded connect probe. A live endpoint is never unlinked. Only a
socket owned by the same UID that returns `ECONNREFUSED` may be classified
stale. Immediately before `unlinkat`, a fresh no-follow `fstatat` MUST match the
recorded device, inode, owner, and type; a missing or changed entry fails
closed. Every parent path component is opened without following symlinks.

The daemon binds a randomized private basename in the trusted directory,
opens that socket inode with `O_PATH|O_NOFOLLOW|O_CLOEXEC`, restricts the pinned
inode to mode 0600 through `/proc/self/fd`, validates it again, and calls
`listen` before publication. It publishes the already-listening inode by an
atomic same-directory `linkat` which MUST fail rather than replace an existing
endpoint, then removes the temporary name. The absolute parent, lock pathname,
pinned inode, and published pathname are revalidated before readiness; the
absolute parent check occurs after published-path validation as the last
namespace check before readiness. The daemon retains the inode pin for the
listener lifetime. If publication, validation, or alias removal fails, it
retains the pin and every possibly live alias name and retries exact-match
cleanup while still holding the lock during shutdown. Each cleanup invocation
makes four immediate attempts for ordinary system errors. If all four fail, it
retains the pin, trusted directory descriptor, lock, inode identity, alias name,
and underlying errno so a later shutdown or destructor invocation can retry;
process exit may leave the pathname for the next same-UID daemon's verified
stale-endpoint recovery. On shutdown it closes the listener, performs the same
fresh identity checks, and removes only pathnames which still identify the
pinned socket. An alias inspection error retains the underlying errno; only an
observed identity mismatch is reported as `ESTALE`.

Given the trusted-ancestor requirement, the 0700 parent directory protects its
contents from other UIDs. Processes running under the same effective UID are
trusted to honor the endpoint lock; Linux has no pathname unlink operation
conditional on a caller-supplied inode, so v1 does not claim protection from a
malicious same-UID process which can rewrite directory entries between an
identity check and `unlinkat`. Atomic no-replace publication prevents
cooperating daemons and bind-time races from adopting or overwriting another
endpoint. Failure to create a socket hard link fails closed before readiness.
The listener, accepted socket, lock descriptor, and any cancellation descriptor
MUST be close-on-exec. Sockets MUST be nonblocking. Sends MUST use
`MSG_NOSIGNAL`.

## 10. Deadline, cancellation, and reconnect boundary

Opening the endpoint, connecting, transmitting HELLO, and validating ACK share
one absolute `CLOCK_MONOTONIC` deadline. The caller MAY supply that deadline as
absolute nanoseconds; otherwise, after synchronous argument and option
validation and before randomness or transport I/O, the runtime converts its
relative timeout to an absolute deadline exactly once. CP-0004 performs one
attempt and no automatic retries or backoff. Every `EINTR` recomputes the
remaining interval from that same deadline. A caller MAY supply a pollable
cancellation FD which remains caller-owned and open for the duration of the call; readable,
hangup, or error readiness cancels the open without consuming the FD. The FD
MUST have `FD_CLOEXEC`. If expiry and cancellation are observed together,
expiry takes precedence. Expiry or caller cancellation closes the socket and
establishes no session.

The daemon applies a bounded absolute monotonic deadline to an accepted client
that has not completed HELLO. gem5 handling MUST use nonblocking readiness and
must not block its event queue. Disconnect does not authorize request replay.
Reconnect, FD passing, shared-memory payloads, eventfd doorbells, data-plane
credits, cancellation messages, and daemon restart recovery require future
versioned capabilities and are outside protocol 1.0.

## 11. Golden handshake

The canonical golden identity uses request ID `0123456789abcdef`, daemon UUID
`00112233445566778899aabbccddeeff`, epoch `0102030405060708`, job UUID
`102132435465768798a9bacbdcedfe0f`, rank 3 of world size 8, and maximum record
65,536. The successful HELLO is 208 bytes with CRC32C `508ae012`; its ACK is
192 bytes with connection ID `1122334455667788` and CRC32C `c09c2612`.

The exact frame bytes and fixed nonces are normative in
`protocol/host-transport-v1.json` and are independently reconstructed by
`tests/test_host_transport_protocol.py`.
