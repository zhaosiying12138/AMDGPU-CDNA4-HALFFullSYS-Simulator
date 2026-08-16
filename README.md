# self-amdgpu-runtime

`self-amdgpu-runtime` is the standalone host-side runtime boundary for the
AMDGPU gem5 simulation stack. It is designed to replace the host-facing
ROCm UMD/KMD submission path while preserving a stable C ABI for compiler,
HIP/OpenCL, Triton, PyTorch, and vLLM integrations.

The current implementation establishes a versioned control connection to one
explicit gem5 daemon endpoint. It implements host-transport handshake 1.0 over
Linux `AF_UNIX` `SOCK_SEQPACKET`, including peer credentials, identity and
capability negotiation, CRC32C framing, cancellation, and one absolute
monotonic deadline. Queue-control protocol 1.0 adds bounded queue lifecycle,
control-only doorbell notification, and asynchronous completion matching.
Simulated-memory protocol 1.0 adds bridge-owned allocation and real byte
transfer through sealed `memfd` staging passed with `SCM_RIGHTS`.
Signal-event protocol 1.0 adds generation-safe signed signals and retryable
waits. Pinned-dispatch protocol 1.0 admits one fixed gfx950 wave64 XOR fixture
without exposing its daemon-owned packet or code image. It does **not** accept
raw AQL, HSACO, kernargs, GPU pointers, or file descriptors as dispatch inputs,
provide a general kernel ABI, model SDMA timing, reconnect, or expose
HIP/HSA/OpenCL operations.

## CP-0009 provider boundary

`<self_amdgpu_runtime/provider.h>` exposes the narrow GemSim provider boundary
used while the pinned ROCr/libhsakmt ABI is being brought up. It publishes the
source-union ThunkLoader manifest (124 entries, 113 HSA and 11 DRM), the
Linux-effective shared/direct counts (123/122), all 17 recorded packed-layout
records, the model interface 1.1 metadata, and the source HSAKMT status table.
The authority identity is pinned by
`protocol/host-transport-v1-provider.json` and its SHA-256 is available through
`sagr_provider_authority_sha256()`.

The layout records expose only the authority's recorded key offsets. Their
`field_count` values count those recorded entries, not necessarily every
member in the upstream C struct; omitted fields are intentionally unspecified
and must not be inferred from this boundary.

`sagr_provider_open()` wraps exactly one existing GemSim transport handshake;
`sagr_provider_get_info()` and `sagr_provider_query_lifecycle()` expose the
negotiated daemon identity, and `sagr_provider_close()` releases the owned
transport. `sagr_provider_invoke()` validates symbol/index and argument
carriers and returns `HSAKMT_STATUS_NOT_SUPPORTED` for operations outside this
checkpoint. It does not export production symbol names, serialize raw pointers
or descriptors, load a host GPU library, or probe a device node. This is an
ABI inventory and provider boundary only, not a generic ROCr/HIP/OpenCL,
code-object, Triton, PyTorch, or vLLM implementation.

## ABI surface

The installed `<self_amdgpu_runtime/runtime.h>` header exposes:

- a fixed-width `sagr_status_t` status type and stable status constants;
- `sagr_abi_version()` for runtime ABI negotiation;
- `sagr_version_string()` for diagnostic version reporting; and
- `sagr_status_string()` for total, non-null status diagnostics;
- size-tagged `sagr_instance_open_options_t`, `sagr_instance_info_t`, and
  `sagr_error_info_t` structures containing fixed-width fields only; and
- `sagr_instance_open()`, `sagr_instance_get_info()`, and
  `sagr_instance_close()` for an independently owned transport instance; and
- size-tagged queue options/results plus create, ring, wait, and destroy APIs;
  and
- size-tagged memory options/info plus allocate, query, H2D, D2H, and free APIs.
- size-tagged signal options/results plus create, load, store, wait, and destroy
  APIs; and
- pinned-dispatch options, admission tickets, completion results, and separate
  submit/wait APIs.

The transport API is ABI version `1.8`; the project release is `0.8.0`.
Status values 0 through 3 retain their original meanings. Transport statuses
are appended, and native `errno` values are diagnostic rather than primary.

`sagr_instance_open_options_init()` selects protocol 1.0, a five-second
relative deadline, no cancellation FD, and mandatory topology capability bit
0. `absolute_deadline_ns` may instead carry an absolute `CLOCK_MONOTONIC`
deadline. A nonnegative `cancel_fd` with `FD_CLOEXEC` is observed but never
consumed or closed; readability, hangup, or error cancels the open, and the
caller must keep the FD open until the call returns. An expired deadline wins
over simultaneous cancellation readiness. UUID, epoch, rank, and world
expectations default to the protocol wildcard tuple. A supervisor should set
an exact nonzero job UUID and epoch plus rank and world size; endpoint
selection alone is not an identity assertion. Opens perform one attempt with
no endpoint scan, retry, or stale-path removal.

Queue control requires capability bit 1 to be both offered and required.
Instances are caller-serialized and support at most eight active queues and
eight accepted, incomplete commands across the daemon session. Queue depth is
bounded to 1 through 64. Command kinds 0 and 1 complete successfully; command
kind 2 is a deterministic asynchronous error-path test. These commands carry
metadata only. A wait timeout or cancellation is retryable. An indeterminate
ACK or a noncanonical frame poisons the queue transport, after which the caller
closes the instance. Queue handles have unique ownership; copied aliases must
not be used after destroy or instance close.

Simulated memory requires capability bit 2 to be both offered and required.
The v1 reference limits are 1024 live allocation slots, 2 GiB per allocation,
4 GiB of total live requested bytes, and 16 MiB per transfer. Allocation is
bridge-owned and sparse on the daemon side; a new or reused allocation reads
as zero. Its returned simulated VA is not a host pointer or a general public
packet address; CP-0008 may bind it only through the pinned daemon-owned
fixture contract.

Memory operations are synchronous and share the instance request-ID sequence
with queue control. H2D sends one exact-size, mode-0600 memfd with
`F_SEAL_SHRINK|F_SEAL_GROW|F_SEAL_WRITE|F_SEAL_SEAL`; D2H sends one mode-0600
`O_RDWR` memfd initially sealed against shrink/grow. A successful daemon D2H
adds the write and seal seals and returns the immutable content CRC32C. The
runtime validates exact size, mode including special bits, access, seals, and
CRC into private scratch before changing the caller buffer.

A failure after a complete request send but before a canonical ACK poisons the
shared transport. A canonical non-OK ACK is determinate and retryable. After a
successful D2H ACK, observable carrier or CRC mismatches poison, while local
scratch allocation or `pread` failure and deadline/cancellation leave the
caller buffer untouched and the known session reusable. Allocation handles
have unique ownership; copied aliases must not be used after free or instance
close.

Signal events require capability bit 3 to be both offered and required. A wait
has an admission ACK and a later completion. Timeout or cancellation after the
ACK retains the same local wait and does not resend it. Signal IDs may be
reused only with a strictly newer generation, and wait, store, queue, memory,
and dispatch records share one strictly increasing request-ID sequence.

Pinned dispatch requires capability bit 4 and selected queue, memory, and
signal capabilities. CP-0008 exposes only fixture
`gfx950-xor-u8-v1`: separate 64-byte input/output allocations, one wave64
workgroup, and `output[lane] = input[lane] ^ 0x5a`. Admission requires a live
signed-one signal with exactly one already-admitted unsatisfied EQ-zero wait.
`sagr_queue_submit_pinned_dispatch()` publishes an immutable ticket only after
a canonical ACK, including request ID, resource generations, trace ID, bound
VAs, packet CRC, and admission tick. `sagr_queue_wait_pinned_dispatch()` uses a
new deadline on every call; timeout or cancellation retains the ticket and
never resends the request.

The generation-valid CP-0007 signal completion at retire tick plus one must be
received before the matching dispatch completion. Both records must agree with
the ticket before the runtime permits D2H or resource cleanup. A canonical
non-OK admission ACK is determinate and reusable; an indeterminate ACK or any
malformed, foreign, reordered, CRC-inconsistent, or tick-inconsistent record
poisons the shared session. Runtime mock tests validate this contract but are
not evidence of successful execution on a real or simulated GPU pipeline.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The `sagr-triton-hsaco-probe` tool checks the compiler-to-runtime boundary for
one HSACO image and kernel name. It validates AMDHSA metadata and reports the
kernel and kernarg layout, but deliberately reports `execution_supported` as
false until PT_LOAD mapping, AQL construction, and GPU dispatch are implemented.

```sh
sagr-triton-hsaco-probe /path/to/kernel.hsaco vecadd
```

The build directory is ignored by Git. Both static and shared builds are
supported through CMake's standard `BUILD_SHARED_LIBS` option. Normal test
builds also install the package and compile an independent consumer with the
default toolchain. Keep `SELF_AMDGPU_RUNTIME_TEST_INSTALLED_PACKAGE=ON` (the
default) for release and packaging validation.

AddressSanitizer/UndefinedBehaviorSanitizer builds are internal verification
artifacts, not directly consumable release packages. In particular, Clang does
not normally record its AddressSanitizer runtime as a dependency of a shared
library; the final executable must be linked with matching sanitizer flags.
Run that matrix explicitly as follows:

```sh
cmake -S . -B build/sanitize -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_C_COMPILER=clang \
  '-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DSELF_AMDGPU_RUNTIME_TEST_INSTALLED_PACKAGE=OFF
cmake --build build/sanitize --parallel
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
  ctest --test-dir build/sanitize --output-on-failure
```

Disabling the installed-package test must remain an explicit matrix choice;
the build does not infer it from compiler flags. Validate installability and a
clean downstream consumer in a separate unsanitized static and shared build.

## Handshake diagnostic

The installed `sagr-handshake` tool opens one explicit endpoint and prints a
single JSON object. It is intended for runtime-to-gem5 conformance checks.

```sh
sagr-handshake \
  --endpoint /absolute/path/to/gemsim.sock \
  --expected-daemon-uuid 00112233445566778899aabbccddeeff \
  --expected-job-uuid 102132435465768798a9bacbdcedfe0f \
  --expected-epoch 0x0102030405060708 \
  --expected-rank 3 \
  --expected-world 8 \
  --timeout-ms 5000
```

For the simulated-memory gate, `--memory-bytes N` automatically offers and
requires capability bit 2. The tool allocates, verifies the initial zero
contents, performs a deterministic H2D/D2H byte and CRC roundtrip, and frees
the allocation. `--memory-alignment` selects 4096 or 65536. Adding
`--memory-reuse` proves that the same slot and VA receive a changed nonzero
generation, are zero initialized again, and are freed. The stable success JSON
contains the `memory` object and nests the optional `reuse` object within it.

```sh
sagr-handshake \
  --endpoint /absolute/path/to/gemsim.sock \
  --memory-bytes 1048576 \
  --memory-alignment 65536 \
  --memory-reuse \
  --timeout-ms 5000
```

Protocol rejection harnesses can override the offered version range with
`--min-version MAJOR.MINOR` and `--max-version MAJOR.MINOR`, and may repeat
`--offer-cap-bit N` or `--require-cap-bit N` for bits 0 through 255. A required
bit must also be offered. `--hold-ms N` flushes the successful JSON result and
keeps the established connection open for that monotonic interval before local
close, allowing another client to exercise the daemon's BUSY path.

For an end-to-end queue-control gate, all three queue options are required.
They automatically offer and require capability bit 1. The tool runs create,
ring/wait for every doorbell, and destroy, then adds queue IDs, generations,
sequences, and completion status to the JSON result:

```sh
sagr-handshake \
  --endpoint /absolute/path/to/gemsim.sock \
  --queue-depth 4 \
  --doorbells 2 \
  --command-kind 1 \
  --timeout-ms 5000
```

For the pinned CP-0008 gate, `--dispatch-fixture gfx950-xor-u8-v1`
automatically offers and requires capability bits 1 through 4. The tool creates
the queue, allocations, and signed-one signal; writes the exact input and zero
output sentinel; arms the EQ-zero wait; submits once; proves a cancelled
post-ACK dispatch wait is retryable; consumes both completion records; verifies
the exact 64-byte D2H result; and performs strict cleanup. Its JSON includes the
four exact byte strings, ticket and completion echo fields, request/trace IDs,
all timing fields, signal completion tick, and retry evidence:

```sh
sagr-handshake \
  --endpoint /absolute/path/to/gemsim.sock \
  --dispatch-fixture gfx950-xor-u8-v1 \
  --timeout-ms 5000
```

## Install and consume

```sh
cmake --install build --prefix "$PWD/install"
```

Consumers can use the exported package without depending on this source tree:

```cmake
find_package(SelfAmdgpuRuntime 0.6 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE SelfAmdgpuRuntime::runtime)
```

## License

This project is licensed under `GPL-3.0-or-later`. See `LICENSE`.
