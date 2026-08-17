# Host-Native AQL Failure Capsule

`tools/aql_failure_capsule.py` freezes the queue head that cannot retire after a
host-native ROCr/gem5 failure. It is generic: selection uses only KMT allocation
identity/ranges, AQL and MQD fields, queue indices, and process/file identity.
It does not inspect a framework, model, operator, kernel name, tensor role,
image hash, expected output, fixed PC, or ticket number.

The v1 capsule is a diagnostic input for a short reproducer. It is deliberately
not an executable packet replay claim.

## Capture contract

Start the unchanged-upstream engine with the existing product environment and
enable the model DSO's generic KMT trace:

```bash
export SAGR_HSAKMT_MODEL_TRACE=1
```

The launcher must retain the ROCr producer process until gem5 has exited after
the panic or nonzero result. Before reaping that producer, run:

```bash
python3 tools/aql_failure_capsule.py capture \
  --pid "$worker_pid" \
  --gem5-pid "$gem5_pid" \
  --worker-log "$run_dir/worker.log" \
  --dispatch-trace "$run_dir/dispatch-trace.jsonl" \
  --log gem5="$run_dir/gem5.log" \
  --freeze-target \
  --output "$run_dir/failure-capsule"
```

`--gem5-pid` is required and that PID must be absent before and after capture.
This is the v1 quiescence boundary: a live gem5 can still mutate the shared
memfd and cannot produce an atomic host-only snapshot. `--freeze-target` uses a
Linux pidfd to send `SIGSTOP` to the exact producer identity and sends
`SIGCONT` only when this invocation stopped it. A stop timeout is also restored;
an already stopped process is left stopped. The process start time, stopped
state, and all queue frontiers are sampled before and after capture; identity,
state, or frontier movement rejects the capsule. Run the tool inside the same
AgentENV PID namespace. Userptr capture also requires the normal Linux
process-memory permission, so the launcher should invoke the tool as the
producer's parent or with an equivalent scoped permission.

The shared KMT backing memfd is discovered from `/proc/$pid/fd`. Use
`--backing-fd N` only when more than one matching memfd is open. If more than one
queue has pending packets, selection is intentionally ambiguous and
`--queue-id N` is required.

Verify a completed capsule without gem5 or ROCm:

```bash
python3 tools/aql_failure_capsule.py verify "$run_dir/failure-capsule"
```

Publication uses Linux `renameat2(RENAME_NOREPLACE)`. The output must be absent;
an existing directory is never updated or replaced.

## Queue-head selection

The worker trace supplies the owner-scoped allocation ledger and queue layout:

- successful `ALLOC_MEMORY_OF_GPU` records provide handle, GPU VA, byte range,
  flags, and shared-backing or userptr offset;
- successful `CREATE_QUEUE`/`UPDATE_QUEUE`/`DESTROY_QUEUE` records provide the
  live ring and read/write pointer addresses;
- doorbell and retirement records preserve the producer/consumer sequence.

The tool reads four distinct 64-bit frontiers. MQD read is the producer's last
synchronized completion and MQD write includes reserved slots. The memfd
doorbell establishes the published tail as `doorbell + 1`; the paired bridge
completion slot is the authoritative retired head even when the stopped
producer has not copied it into MQD read. The selected packet is therefore the
exact first unretired published packet after gem5 exits. Packet `head + 1`, when
still below the published tail, is frozen separately as the next packet. A
reserved but not doorbelled slot is never selected.

The ring address is calculated modulo queue depth. Regressed or inconsistent
frontiers, a pending span larger than the ring, no pending queue, or multiple
pending queues without explicit selection fails closed. V1 fully decodes only
a kernel-dispatch head; another packet type at the head fails closed. A
non-kernel next packet is still preserved byte-for-byte with its common type
metadata.

The current ioctl trace cannot expose the provider's allocation generation and
does not include a handle on successful frees. Its raw allocation handle, log
ordinal, and line number are therefore recorded, while generation is explicitly
`null`. When the log retains older mappings, the newest allocation containing a
VA is recorded as an identity candidate under the bridge invariant that live
owner mappings cannot overlap. This is not a provider-liveness proof, and no
unavailable generation is invented. Aligned kernarg qwords that fall in an
observed allocation are likewise conservative pointer candidates, not typed
kernarg claims.

## Frozen data

The canonical manifest and its SHA-256 bind:

- failing and next 64-byte AQL packet, decoded fields, SHA-256, and CRC32C;
- 64-byte Code Object V3/common descriptor or complete 256-byte versioned Code
  Object V2 descriptor, entry relation, and complete resident code allocation
  bytes;
- exact CP kernarg read span (descriptor size, otherwise preload end), the
  one-byte residency probe when that span is empty, and a 64-byte completion
  signal;
- 256-byte MQD, ring/read/write identity, pending range, scratch range, and
  inactive-signal address;
- allocation registry metadata and full-range SHA-256 for every resource found
  through packet fields and MQD fields, plus conservative aligned-kernarg
  pointer candidates;
- complete stable prefixes of worker, dispatch, and requested logs, including
  the first panic/fatal/traceback marker;
- process maps, status, stat, command line, executable link, fd targets, and
  fdinfo. Process environment and arbitrary fd contents are not captured.

Application and scratch allocations are hashed but not copied by default.
Resident code is copied because descriptor and entry bytes must remain usable
for a focused reproducer. `--max-hash-bytes`, `--max-code-bytes`,
`--max-kernarg-bytes`, per-file `--max-log-bytes`, and
`--max-total-log-bytes` are explicit fail-closed resource budgets, not
truncation controls.

## Replay boundary

Submitting a saved 64-byte packet alone is incorrect. Its addresses depend on
owner/generation-bound allocations, code residency, kernarg and signal state,
MQD indices, scratch, and completion ordering. A panic during CU execution also
depends on simulator event-queue and pipeline state that this host-side capsule
does not serialize.

For that reason every v1 manifest contains `replay.eligible=false` and exact
blockers. The capsule is sufficient to construct and validate a narrow generic
reproducer, compare descriptor/code/kernarg/allocation identities across a fix,
and decide whether a gem5 checkpoint or a new resource-reinstantiation protocol
is needed. It must not be relabeled as packet replay evidence.

## Host-only gate

```bash
python3 -m unittest tests.test_aql_failure_capsule -v
```

The 22 host-only cases cover doorbell/completion selection, lagging MQD read,
reserved-but-unpublished slots, kernel and non-kernel packet handling,
descriptor/code/kernarg/signal/MQD capture, V2/V3 descriptor ABI, whole-range
scratch validation, resource budgets, pidfd stop/restore, canonical semantic
verification, artifact tamper rejection, absent-only output, exited-gem5
enforcement, and ambiguous multi-queue rejection. They launch no gem5 process.
