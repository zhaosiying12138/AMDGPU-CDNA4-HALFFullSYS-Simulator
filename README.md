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
It does **not** submit GPU packets or kernels, allocate GPU memory, pass
descriptors or host pointers, reconnect, or expose HIP/HSA/OpenCL operations.

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
- size-tagged queue options/results plus create, ring, wait, and destroy APIs.

The transport API is ABI version `1.2`; the project release is `0.3.0`.
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

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
ctest --test-dir build --output-on-failure
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

## Install and consume

```sh
cmake --install build --prefix "$PWD/install"
```

Consumers can use the exported package without depending on this source tree:

```cmake
find_package(SelfAmdgpuRuntime 0.3 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE SelfAmdgpuRuntime::runtime)
```

## License

This project is licensed under `GPL-3.0-or-later`. See `LICENSE`.
