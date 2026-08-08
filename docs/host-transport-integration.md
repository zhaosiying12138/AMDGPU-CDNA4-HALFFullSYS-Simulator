# CP-0004 Through CP-0006 Host Transport Integration

`scripts/test_host_transport_integration.py` is the real cross-process gate for
the frozen host transport 1.0 handshake, its bounded queue-control extension,
and the bridge-owned simulated-memory data path.
It starts the supplied gem5 binary with the `HostGPUBridge` SimObject and invokes
the supplied `sagr-handshake` runtime CLI against its pathname `SOCK_SEQPACKET`
endpoints.

The gate does not build either child and does not download anything. Build the
two binaries separately, then run:

```sh
python3 scripts/test_host_transport_integration.py \
  --gem5 projects/gem5/build/VEGA_X86/gem5.opt \
  --runtime-cli projects/self-amdgpu-runtime/build/sagr-handshake \
  --keep-work-dir
```

The script emits one machine-readable JSON summary. A successful run covers:

- independent Python wire construction driven by
  `protocol/host-transport-v1.json`, including CRC-32C and a dynamic HELLO that
  must equal the canonical golden frame byte for byte;
- a correlatable, structurally malformed HELLO receiving the canonical
  `MALFORMED` ACK before a fresh valid connection succeeds;
- rejection of ordinary `SCM_RIGHTS`, a zero-length record, a zero-length
  record carrying `SCM_RIGHTS`, and ancillary control truncation, with the
  gem5 `/proc/<pid>/fd` count returning exactly to its pre-injection baseline
  after every case and a subsequent handshake succeeding;
- closure of a connected peer which sends no HELLO at the configured daemon
  handshake deadline;
- an OK handshake followed on the same connection by either a structurally
  valid unsupported-version HELLO or, against an independent daemon, a
  malformed-role HELLO; these must receive `PROTOCOL_STATE` and `MALFORMED`
  respectively before the connection closes;
- eight accepted stalled clients followed by a ninth, valid HELLO receiving
  `RESOURCE_EXHAUSTED` during the bridge's blocking startup phase, then a second
  eight-plus-one phase in steady state where the ninth connection remains open
  and idle for 100 ms before its delayed HELLO receives the canonical rejection;
  this proves the bounded rejection slots use normal readiness, deadline, and
  nonblocking ACK machinery rather than an accept-time one-shot read;
- version, capability, daemon-instance, and topology rejection followed by a
  successful handshake against the same daemon;
- queue capability negotiation followed by CREATE, three strictly ordered
  doorbells, their correlated asynchronous completions, and DESTROY;
- a separate queue session whose accepted control-error command completes with
  the canonical deterministic `INTERNAL` status and error code;
- a memory-capable session that allocates a three-chunk sparse range, verifies
  initial zero-fill, performs real sealed-memfd H2D and D2H transfer, compares
  bytes and CRC, frees the allocation, reuses its slot with a new generation,
  verifies zero-fill again, and frees the reused allocation;
- one established runtime holding the daemon while a second runtime receives
  `BUSY`;
- independent daemon endpoints and exact identities for world sizes 1, 2, 3,
  4, and 8, including exact gem5 peer-PID routing;
- runtime deadline expiry and EOF when an actual bridge process is paused or
  terminated;
- recovery from the stale socket left by a killed bridge, preservation of an
  unrelated live socket, and failure when gem5 listeners are disabled;
- rejection of unsafe or symlinked parents, non-socket or symlink endpoints,
  unsafe lock files, and a second gem5 competing for an active endpoint;
- inode-gated shutdown cleanup which preserves a same-UID replacement socket;
  and
- successful ACK fields, endpoint cleanup, and retained 0600 lock files.

Every daemon has a separate gem5 output directory and combined process log.
Raw socket operations are recorded in `raw-wire.jsonl` in the same run
directory. The raw codec does not import or invoke either child implementation;
the checked-in protocol JSON and golden frames are its only wire-format input.
The run directory and endpoint parent are mode 0700. At most eight gem5
processes run concurrently and the resource-exhaustion phase opens at most nine
clients against one daemon. Each child is placed in its own process group, and
all subprocess and socket operations have wall-clock deadlines. The script sends
`SIGTERM` and then `SIGKILL` to any remaining process groups during cleanup.
Failed runs always retain their run directory; successful runs retain it only
with `--keep-work-dir`.

The fixture suite exercises the JSON-driven encoder/decoder against both
canonical golden frames, the CRC-32C check vector, Linux zero-length descriptor
passing and `MSG_CTRUNC` behavior, and bounded raw receive timeout without
starting gem5:

```sh
python3 -m unittest discover -s tests -p 'test_host_transport_*.py' -v
```

This gate proves framing, negotiation, identity, endpoint lifecycle, deadline
behavior, the bounded control-queue/event path, and functional bytes stored by
the gem5 bridge. It does not prove packet-visible GPU VRAM, SDMA timing, packet
submission, code object execution, collectives, or any GPU computation.
