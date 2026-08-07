# self-amdgpu-runtime

`self-amdgpu-runtime` is the standalone host-side runtime boundary for the
AMDGPU gem5 simulation stack. It is designed to replace the host-facing
ROCm UMD/KMD submission path while preserving a stable C ABI for compiler,
HIP/OpenCL, Triton, PyTorch, and vLLM integrations.

This baseline intentionally contains only the version and status foundation.
It does **not** claim to submit packets, manage doorbells, allocate GPU memory,
load code objects, or communicate with gem5. Those features require explicit
protocol designs and conformance tests against the pinned ROCm and gem5 source
baselines; returning success before they exist would make failures unsafe and
non-diagnostic.

## ABI surface

The installed `<self_amdgpu_runtime/runtime.h>` header exposes:

- a fixed-width `sagr_status_t` status type and stable status constants;
- `sagr_abi_version()` for runtime ABI negotiation;
- `sagr_version_string()` for diagnostic version reporting; and
- `sagr_status_string()` for total, non-null status diagnostics.

The public ABI begins at version `1.0`; the project release is `0.1.0`.
New transport APIs will be added only after their ownership, lifetime,
concurrency, error, and wire-protocol contracts are specified.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The build directory is ignored by Git. Both static and shared builds are
supported through CMake's standard `BUILD_SHARED_LIBS` option.

## Install and consume

```sh
cmake --install build --prefix "$PWD/install"
```

Consumers can use the exported package without depending on this source tree:

```cmake
find_package(SelfAmdgpuRuntime 0.1 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE SelfAmdgpuRuntime::runtime)
```

## License

This project is licensed under `GPL-3.0-or-later`. See `LICENSE`.
