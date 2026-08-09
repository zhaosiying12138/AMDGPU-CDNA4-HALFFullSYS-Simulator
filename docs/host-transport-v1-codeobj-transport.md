# Fixed-Record HSACO Transport

This document explains the CP-0013 A1 authority in
[`protocol/host-transport-v1-codeobj-transport.json`](../protocol/host-transport-v1-codeobj-transport.json).
It extends the base host transport without changing the source-only
[`CODE_OBJECT_ABI_V1`](../protocol/host-transport-v1-codeobj.json) authority or
the CP-0008 pinned-dispatch records.

## A1 boundary

A1 is transport staging only. It proves that one authority-conforming gfx950
HSACO image can be copied into daemon-owned memory, correlated with a
connection-scoped ID/generation, and bound to a SHA-256 digest. It does not map
`PT_LOAD` ranges, create process page mappings, apply relocations, publish
kernarg bytes, construct AQL, submit a queue packet, execute gem5 instructions,
or report an executable ISA.

Consequently, every A1 ACK has zero `mapped_base_va`, `descriptor_va`,
`code_va`, and `kernarg_va`, including a successful COMMIT. The u64 object ID
and generation are protocol identities, not pointers or GPU addresses.

Future gates are deliberately separate:

1. A2 independently validates and atomically maps `PT_LOAD` ranges, applies
   required relocations, and zero-fills `memory_size - file_size`.
2. A3 materializes complete explicit and hidden kernarg bytes and a
   daemon-owned AQL packet from validated handles.
3. A4 submits through the real HSA queue, fetch, decoder, and retirement path
   and retains output and signal evidence.

## Capability and message IDs

`CODE_OBJECT_TRANSPORT_V1` is capability bit 7, byte 0, mask `0x80`. It is
selected only when both offered and required. Bit 6 (`0x40`) remains the
source-only `CODE_OBJECT_ABI_V1` reservation and is not repurposed.

Existing wire IDs 1 through 15 remain unchanged. A1 allocates only:

| Name | ID |
| --- | ---: |
| `CODE_OBJECT_REQUEST` | 16 |
| `CODE_OBJECT_ACK` | 17 |

The request opcode is `BEGIN=1`, `CHUNK=2`, or `COMMIT=3`. Each request consumes
a new nonzero, strictly increasing base-header request ID and receives exactly
one correlated ACK.

Message IDs 18 (`KERNEL_DISPATCH_REQUEST`), 19 (`KERNEL_DISPATCH_ACK`), and
20 (`KERNEL_DISPATCH_COMPLETION`) are reserved for a separately reviewed A2/A3
dispatch extension; A1 assigns them no payload or execution semantics.

## Fixed record

Every request and ACK is exactly 4096 bytes: the existing 80-byte big-endian
header plus a 4016-byte payload. This fits a connection whose negotiated
`max_record` is the base minimum 4096. The base frame CRC-32C covers all 4096
bytes with header bytes 64 through 67 treated as zero.

The first 48 payload bytes are common to every request:

| Offset | Bytes | Field | Rule |
| ---: | ---: | --- | --- |
| 0 | 2 | `major` | 1 |
| 2 | 2 | `minor` | 0 |
| 4 | 2 | `opcode` | 1, 2, or 3 |
| 6 | 2 | `flags` | zero |
| 8 | 8 | `object_id` | zero on BEGIN; exact live ID otherwise |
| 16 | 8 | `object_generation` | zero on BEGIN; exact live generation otherwise |
| 24 | 8 | `image_offset` | opcode-specific |
| 32 | 4 | `byte_count` | opcode-specific |
| 36 | 4 | `chunk_index` | opcode-specific |
| 40 | 4 | `chunk_crc32c` | CHUNK copied bytes only; zero otherwise |
| 44 | 4 | `reserved` | zero |

All integers in the envelope are big-endian. The copied ELF and its 64-byte
kernel descriptor remain opaque little-endian bytes and are validated as such.

No request or ACK accepts or returns an ancillary descriptor. `MSG_TRUNC`,
`MSG_CTRUNC`, any `SCM_RIGHTS`, or any other ancillary data closes the
connection before code-object state changes. No field is a host pointer,
pointer-sized ABI value, function address, FD, dma-buf, or render-node handle.

## BEGIN manifest

BEGIN has zero object identity, offset, count, index, and chunk CRC. Its fixed
manifest occupies payload offsets 48 through 1311; bytes 1312 through 4015 are
zero.

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 48 | 8 | `image_size`, 1 through 64 MiB |
| 56 | 4 | `chunk_data_bytes`, exactly 3968 |
| 60 | 4 | `chunk_count`, `ceil(image_size / 3968)` |
| 64 | 4 | `segment_count`, 1 through 16 |
| 68 | 4 | selected `kernel_index` |
| 72 | 32 | whole-image SHA-256 |
| 104 | 2 | ELF machine, 224 (`EM_AMDGPU`) |
| 106 | 2 | ELF type, 3 (`ET_DYN`) |
| 108 | 1 | ELF OSABI, 64 |
| 109 | 1 | ELF ABI version, 2 through 4 |
| 110 | 2 | reserved, zero |
| 112 | 4 | ELF flags, low byte `0x4f` |
| 116 | 4 | gfx target, 950 |
| 120 | 4 | code-object version, 4 through 6 |
| 124 | 4 | metadata major, 1 |
| 128 | 4 | metadata minor, 1 or 2 |
| 132 | 4 | relocation count, zero in A1 |
| 136 | 4 | kernarg segment size |
| 140 | 4 | kernarg alignment, nonzero power of two |
| 144 | 4 | fixed group segment bytes |
| 148 | 4 | fixed private segment bytes |
| 152 | 4 | maximum flat workgroup size |
| 156 | 4 | wavefront size, 64 |
| 160 | 4 | SGPR count |
| 164 | 4 | VGPR count |
| 168 | 4 | dynamic stack, zero in A1 |
| 172 | 4 | descriptor size, 64 |
| 176 | 8 | signed descriptor-to-code entry offset |
| 184 | 8 | ELF-relative code virtual address |
| 192 | 8 | code file offset |
| 200 | 8 | code size |
| 208 | 8 | ELF-relative descriptor virtual address |
| 216 | 8 | descriptor file offset |
| 224 | 128 | NUL-padded kernel name |
| 352 | 128 | NUL-padded descriptor symbol |
| 480 | 64 | exact descriptor bytes |
| 544 | 768 | sixteen 48-byte segment records |
| 1312 | 2704 | reserved, zero |

Each used segment record contains, at relative offsets 0, 4, 8, 16, 24, 32,
and 40: `type` u32, `flags` u32, `file_offset` u64, `virtual_address` u64,
`file_size` u64, `memory_size` u64, and `alignment` u64. Type is `PT_LOAD` (1),
flags are R (4), RX (5), or RW (6), `file_size <= memory_size`, and alignment
is zero or a power of two. Unused records are all zero.

A1 validates file and virtual range arithmetic, image bounds, alignment,
pairwise overlap, descriptor equality, and the relation
`descriptor_address + descriptor_entry_offset == code_address`. It records
those values but maps no range.

## CHUNK and COMMIT

CHUNK copies at most 3968 bytes at payload offset 48. The copied prefix length
is `byte_count`; every remaining byte through payload offset 4015 is zero. The
daemon accepts only the next zero-based index and the next contiguous image
offset. It validates the full-frame CRC, the per-chunk CRC-32C over exactly the
copied bytes, bounds, ownership, generation, and padding before changing
staging or advancing expected state.

COMMIT carries the BEGIN image size in `byte_count`, BEGIN chunk count in
`chunk_index`, zero `image_offset` and chunk CRC, and the repeated whole-image
SHA-256 at payload offset 48. Payload offsets 80 through 4015 are zero. It is
accepted only after every byte and chunk. The daemon recomputes SHA-256 over
exactly `image_size` staged bytes in increasing offset order, compares both
declared digests and the manifest-to-image facts, and then marks staging
immutable. CRC-32C never substitutes for SHA-256 identity.

A malformed post-BEGIN record, stale identity, gap, duplicate, overlap, CRC,
padding, digest, or manifest mismatch aborts and zeroizes the transaction.
Timeout, cancellation, connection loss, or an indeterminate ACK poisons the
session; the client does not retry an uncertain record in place.

## ACK layout

The ACK payload is fixed at 4016 bytes:

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 2 | major, 1 |
| 2 | 2 | minor, 0 |
| 4 | 4 | base wire status |
| 8 | 2 | echoed opcode |
| 10 | 2 | flags, zero |
| 12 | 4 | reserved, zero |
| 16 | 8 | object ID |
| 24 | 8 | object generation |
| 32 | 8 | accepted offset |
| 40 | 4 | accepted count |
| 44 | 4 | chunk index |
| 48 | 8 | mapped base VA, zero in A1 |
| 56 | 8 | descriptor VA, zero in A1 |
| 64 | 8 | code VA, zero in A1 |
| 72 | 8 | kernarg VA, zero in A1 |
| 80 | 8 | image size |
| 88 | 4 | selected kernel index |
| 92 | 4 | segment count |
| 96 | 8 | simulation tick |
| 104 | 32 | image SHA-256 |
| 136 | 4 | error code |
| 140 | 4 | reserved, zero |
| 144 | 3872 | reserved, zero |

BEGIN success is the only operation that creates an ID/generation. CHUNK and
COMMIT ACKs echo that exact pair. A failure publishes no new identity or
address. IDs and generations are connection- and daemon-instance-scoped,
nonzero, monotonic, never wrapped, and changed on slot reuse.

## Source constraints and observed candidate

The runtime parser in
`projects/self-amdgpu-runtime/src/code_object.c` already validates the ELF,
metadata, symbols, descriptor, and up to 16 `PT_LOAD` records, but its dispatch
binding is metadata-only and contains no whole-image digest. The current memory
transport uses memfd/`SCM_RIGHTS`; A1 does not reuse that carrier.

The CP-0012 generated vecadd artifact is an observation, not a tracked A1
acceptance fixture. It is 5672 bytes with SHA-256
`b0e07d4d34826177f7261b706dc9bac139d7a59b5217e8094fa714371874efaf`
and therefore uses two chunks of 3968 and 1704 bytes. Its descriptor is at
virtual/file address `0x8c0`, its entry is at `0x1a00`, and the descriptor's
little-endian signed entry offset is `0x1140`:

```text
0x8c0 + 0x1140 = 0x1a00
```

This correct relation matters for later execution: an AQL `kernel_object`
points to the mapped descriptor, not directly to `.text`. It does not weaken
the A1 no-mapping/no-execution boundary.

## Acceptance

The authority and its independent oracle test freeze field offsets, lengths,
capability/message/opcode IDs, zero-padding, frame/chunk CRC rules, SHA-256
binding, transaction order, ownership, and the A1 zero-address boundary. Child
runtime/gem5 code plus retained integration evidence is required before any
mapping or execution claim.
