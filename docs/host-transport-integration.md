# CP-0004 Through CP-0008 Host Transport Integration

`scripts/test_host_transport_integration.py` is the real cross-process gate for
the frozen host transport 1.0 handshake, its bounded queue-control extension,
the bridge-owned simulated-memory data path, and the generation-safe signal
event path. CP-0008 adds one source-pinned gfx950 AQL dispatch through the real
gem5 HSAPacketProcessor, GPUCommandProcessor, GPUDispatcher, and compute-unit
path.
It starts the supplied gem5 binary with the `HostGPUBridge` SimObject and invokes
the supplied `sagr-handshake` runtime CLI against its pathname `SOCK_SEQPACKET`
endpoints.
The CP-0004 through CP-0007 matrix uses `host_bridge.py`; the one CP-0008
process uses the separate `host_dispatch.py` config so the execution topology
and retained evidence cannot be confused with the control-only fixture.

The gate does not build either child and does not download anything. Build the
two binaries separately, then run:

```sh
python3 scripts/test_host_transport_integration.py \
  --gem5 projects/gem5/build/VEGA_X86/gem5.opt \
  --runtime-cli projects/self-amdgpu-runtime/build/cp8-static/sagr-handshake
```

The default configs are
`projects/gem5/configs/example/gemsim/host_bridge.py` and
`projects/gem5/configs/example/gemsim/host_dispatch.py`. Override the latter
with `--dispatch-gem5-config`. Missing configs and non-executable child
binaries are rejected before any process starts. The dispatch client and
daemon have separate finite wall-clock limits controlled by
`--dispatch-process-timeout-seconds` and
`--dispatch-server-run-timeout-ms`. The runner passes the dispatch trace as the
absolute `--dispatch-trace-path` required by `host_dispatch.py`; it does not
substitute a control-only config or a different trace option.

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
- a signal-capable session that creates a signal at -7, loads it, admits a
  signed `GTE 0` wait, observes a bounded local timeout, stores 42, retries the
  identical wait without sending another request, validates the correlated
  gem5 event-queue completion, loads 42, destroys the signal, and reuses the
  same slot with a new generation before destroying it again;
- a dispatch-capable session invoked as
  `--dispatch-fixture gfx950-xor-u8-v1` which creates one queue, distinct
  64-byte input/output allocations, and a signed-one CP-0007 signal; copies
  the exact input bytes `00..3f`; admits one unsatisfied same-owner `EQ 0`
  wait and observes its initial bounded timeout; submits the pinned dispatch;
  validates the admission ticket before treating any execution field as
  published; performs one deterministic locally canceled dispatch wait and
  retries it without a second request; validates the GPU-updated signal
  completion at packet-retire tick `R + 1` before the dispatch completion;
  and only then performs D2H and accepts the exact `input XOR 0x5a` bytes and
  CRC-32C `0x796671ec`;
- exact cross-correlation of the dispatch request ID, trace ID, queue,
  allocation and signal handles and generations, packet-visible VAs,
  materialized packet CRC, output CRC, and admission/start/end/retire ticks;
- a retained `dispatch-trace.jsonl` with the twelve authority-ordered events,
  pinned fixture/code/AQL hashes, byte-recomputed materialized AQL and kernarg
  hashes, HSAPacketProcessor queue/fetch provenance, GPU task identity,
  one CU-0 wave64 workgroup, an exact 64-byte output-store range, real
  finish-packet retirement, the signed one-to-zero CP-0007 mirror, and the
  final wire summary;
- gem5 statistics proving exactly one fetched packet, command-processor
  submission, dispatcher start/completion, workgroup and wave; positive GPU
  retired-instruction and global-store instruction counts; exactly 64 global
  store bytes; and `host_fallback_count = 0`. The bridge counters are read from
  `system.host_gpu_bridge.cp8_*` (and
  `system.host_gpu_bridge.host_fallback_count`), while real retired
  instructions are read from `system.cpu1.CUs.numInstrExecuted`;
- repeated live `/proc/<pid>/maps` and open-descriptor audits for both gem5 and
  the runtime CLI, plus both complete process logs, rejecting `libhsa*`,
  `libamdhip64`, `/dev/kfd`, and `/dev/dri`;
- successful post-retirement destruction/free of every CP-0005 through
  CP-0007 resource used by the fixture;
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
directory. The dispatch process additionally retains `dispatch-trace.jsonl`
and its gem5 `stats.txt`. After the runtime has validated both completions,
accepted the exact D2H bytes, destroyed/freed all fixture resources, and closed
the session, the dispatch daemon exits through `host_dispatch.py` so its
explicit stats dump completes. The runner waits for that clean exit before it
parses either evidence file and requires exactly one
`cause=host GPU dispatch session complete code=0` exit marker naming that
daemon's `stats.txt`; a wall timeout or any other cause fails the gate. It never
stops gem5 early on the successful path.
The raw codec does not import or invoke either child implementation;
the checked-in protocol JSON and golden frames are its only wire-format input.
The run directory and endpoint parent are mode 0700. At most eight gem5
processes run concurrently and the resource-exhaustion phase opens at most nine
clients against one daemon. Each child is placed in its own process group, and
all subprocess and socket operations have wall-clock deadlines. The script sends
`SIGTERM` and then `SIGKILL` to any remaining process groups during cleanup.
Starting with CP-0008, successful and failed run directories are retained
unconditionally because the dispatch trace, stats, complete process logs, and
raw-wire log are acceptance provenance. The JSON summary exposes the absolute
path as `retained_work_dir`; `--keep-work-dir` remains only as a compatibility
flag.

The fixture suite exercises the JSON-driven encoder/decoder against both
canonical golden frames, the CRC-32C check vector, Linux zero-length descriptor
passing and `MSG_CTRUNC` behavior, and bounded raw receive timeout without
starting gem5:

```sh
python3 -m unittest discover -s tests -p 'test_host_transport_*.py' -v
```

This gate proves framing, negotiation, identity, endpoint lifecycle, deadline
behavior, the bounded control queue/memory/signal paths, and exactly one
source-pinned, traceable gfx950 wave64 dispatch. CP-0008 is not a generic GPU
runtime. It does not prove arbitrary packets, geometry, kernels or code-object
loading; ELF/MsgPack parsing or relocation; ROCr/libhsakmt provider
compatibility; HIP, OpenCL, Triton, PyTorch or vLLM; multi-CU scheduling,
collectives, tensor parallelism, model execution, or performance.

CP-0008 is the accepted P1 dispatch boundary. The next action is
`P2-KMT-ABI-01`: inventory the pinned ROCr ThunkLoader and libhsakmt ABI and
prepare a source-exact provider skeleton. That follow-on gate must audit source
layouts, resolved symbols, status/error behavior, and explicit unsupported
capabilities before making any P2 provider-compatibility claim.
