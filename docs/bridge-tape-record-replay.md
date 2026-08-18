# Bridge tape: record now, replay a point in time later

## Problem

A model kernel defect is reached only after weight load and prefill, tens of
minutes to hours into a run. Today the only way to look at the failing dispatch
again is to repeat the whole run. The retained `dispatch-trace.jsonl` cannot
close that loop: it records descriptors and never bytes. It has the kernel
object VA, the kernarg VA and size, and the grid geometry, but not the code
object, the kernarg contents, or any buffer the kernel reads. Replaying it
would launch a kernel over undefined memory.

The tractable recording point is the bridge wire, not the AQL packet. Every
kernel-visible byte crosses that wire or lives in the one shared backing the
runtime exports across it.

## Increment 1 — the tape (implemented)

`HostGPUBridge` takes an optional `bridge_tape_path`
(`--bridge-tape-path` in `configs/example/gemsim/host_dispatch.py`). When set,
the bridge appends every wire record in both directions to one binary log:
direction, client fd, connection generation, `curTick()`, `CLOCK_MONOTONIC`
nanoseconds, whether an SCM_RIGHTS carrier accompanied the record, the record
length, and the unmodified record bytes.

- Ingress is tapped in `receiveRecords`, immediately before the record is
  pushed onto `pendingRecords`, so the tape holds exactly what the bridge
  admitted.
- Egress is tapped in `flushClient`, after a complete `send`, so the tape holds
  exactly what reached the client. An enqueued record that was never
  transmitted is not a wire event and is not recorded.

The writer is in `src/dev/amdgpu/host_gpu_bridge_tape.{hh,cc}`; the layout is
documented on `HostGPUBridgeTape`. It is a pure observer: it never inspects,
filters, or interprets a record, so it is valid for any kernel, any operator,
and any future protocol revision. `tools/bridge_tape_decode.py` renders a tape
as JSONL, decoding only the frozen 80-byte transport header and the two fixed
operation selectors of the versioned contract.

Acceptance was observational freedom: three product OpenCL vector-add runs
without the tape and three with it produced one identical masked
`dispatch-trace.jsonl` digest, and the three tapes each held 47 records and
74064 record bytes with zero CRC failures.

## What increment 1 still does not give you

The tape is complete for everything that *crosses* the wire. It is not
complete for what the client mutates *behind* the wire. The runtime creates one
sparse memfd per process in `hsakmt_model.c`
(`memfd_create("self-amdgpu-hsakmt-model", ...)`, `SAGR_HSAKMT_MODEL_MEMFD_BYTES`
= 288 GiB declared), hands gem5 a descriptor for it through the KMT
`EXPORT_BACKING` operation, and thereafter writes the AQL ring, kernel
descriptors, machine code, kernargs, completion signals, and every data buffer
into it directly. gem5 reads those bytes with `pread`
(`HostKmtSharedBacking::read` in `host_gpu_kmt_memory.cc`). None of that traffic
is a wire record.

So a tape alone can reproduce the *control* sequence but not the *contents* of
memory at the moment of failure. Increment 2 supplies the contents.

## Increment 2 — sparse backing snapshot (design only)

Add a second optional gem5 parameter, e.g. `bridge_snapshot_path`, plus a
trigger expressed in the terms the bridge already owns: an execution ticket, a
retired-dispatch count, or a KMT queue write index. When the trigger fires,
walk every registered `HostKmtSharedBacking` with `SEEK_DATA`/`SEEK_HOLE` and
write the populated extents, each as `(offset, length, bytes)`, into a snapshot
file that names the tape sequence number it is aligned to.

This is cheap, and measured rather than assumed. On a live SGLang TP1 lane the
288 GiB declared memfd held 31 extents and 263.9 MiB of resident data, and a
full `SEEK_DATA`/`SEEK_HOLE` walk took 2.9-5.8 ms. Snapshotting a few hundred
megabytes at a chosen dispatch is a sub-second operation, not a checkpoint.

Design constraints:

- The trigger must be a transport-level quantity. An execution ticket, a
  retired count, or a queue index is admissible. A kernel name, a code hash, or
  a program counter is not, and would violate the generality rule.
- The snapshot must be taken at a point where the bridge is not mid-record, so
  the natural site is the same place the dispatch trace batch is written.
- A snapshot has to record which client fd and generation owns each backing, so
  replay can rebuild the mapping between recorded owners and fresh ones.
- Doorbell and completion regions live in the reserved tail described by
  `protocol/host-transport-v1-kmt.json`; the snapshot must capture them as
  ordinary extents and must not normalize them, because their exact values are
  what the queue observer resumes from.

## Increment 3 — the replay driver (design only)

A standalone client, modelled on `projects/self-amdgpu-runtime/tools/sagr-handshake.c`,
which already opens the endpoint, negotiates capability bits, and frames
records correctly. It would:

1. Read a tape and a snapshot.
2. Create a memfd of `SAGR_HSAKMT_MODEL_MEMFD_BYTES`, restore the snapshot
   extents into it, and keep every hole a hole.
3. Connect to a fresh gem5 and replay the recorded ingress records for one
   connection in order, rewriting only the fields that a new session must own:
   the daemon instance UUID, connection id, and job epoch in the transport
   header, plus the CRC. Request ids and KMT owner/object handles are chosen by
   the client in the original run and can be replayed verbatim.
4. Substitute its own memfd for the carrier on the recorded `EXPORT_BACKING`
   record, and drop the records that precede the chosen snapshot point once the
   snapshot already encodes their effect.
5. Compare each received egress record against the recorded one, masking the
   same identity fields, and stop at the first divergence or at the requested
   ticket.

Known unresolved points, in the order they will bite:

- **Where a replay may start.** The bridge's KMT state machine has ordering
  requirements (owner open, backing export, queue create, doorbell). A replay
  that starts mid-stream must either replay the full prefix of session-shaping
  records — cheap, since they are small — or reconstruct that state, which is
  harder. Replaying the prefix and snapshotting only memory is the simpler
  correct choice.
- **Identity rewriting.** Every record carries the daemon UUID and connection
  id of the recording run, and the CRC covers them. The driver must recompute
  the CRC after substitution. The tape file header carries the recorded
  identity precisely so this substitution is mechanical.
- **Timing.** The tape stores simulated ticks and host monotonic nanoseconds,
  but a fresh gem5 will not reproduce them. Replay must be ordering-faithful,
  not tick-faithful, and any comparison must therefore mask ticks exactly as
  the observational-freedom gate does.
- **Multiple connections.** A model run has several clients. The tape
  distinguishes them by (fd, generation); a replay driver should target one
  connection at a time and refuse a tape whose target connection interleaves
  with a dependency on another.

## Recording a model run

The tape is currently reachable only through a gem5 command line. A model run
starts gem5 from `projects/self-amdgpu-runtime/src/managed_session.c`, which
builds a fixed argv (search for `--dispatch-trace-path`). Recording a model run
needs two more argv entries there, gated by an environment variable so the
default path is unchanged. That change is deliberately not part of increment 1,
because it requires rebuilding and reinstalling the runtime under a live lane.
