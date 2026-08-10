# Repository-local ROCm environment

The simulator does not install ROCm into `/opt/rocm`, change
`/etc/ld.so.conf`, write `/etc/OpenCL/vendors`, or modify the host Python/Conda
environment. Generated files live below an ignored, versioned prefix under
`env/rocm/`.

This is intentionally a simulator prefix rather than a claim of a full vendor
ROCm installation. It contains the pinned LLVM/Clang/lld toolchain, AMD device
libraries, `self-amdgpu-runtime`, the bounded local `libOpenCL.so.1`, and the
`opencl-vecadd` example. Production `libhsa-runtime64`, `libhsakmt`, and
`libamdhip64` are rejected. `/dev/kfd` and `/dev/dri` are not required or
opened by the accepted route. The simulator runtime replaces those host
KMD/UMD entry points with the GemSim transport. HIP, normal Triton Python, and
general OpenCL compatibility remain later product gates.

## Prepare

Run from the repository root. A full build creates only the repository-local
prefix:

```bash
scripts/setup_rocm_env.sh --all --jobs "$(nproc)"
```

Verify an existing prefix without rebuilding it:

```bash
scripts/setup_rocm_env.sh --verify-only
```

The script locks the `llvm-project`, `rocm-systems`, `gem5`, and
`self-amdgpu-runtime` commits and trees and rejects dirty source checkouts. Its
child CMake, Ninja, compiler, and test processes run under `env -i` with an
explicit `/usr/bin:/bin` bootstrap path and private `HOME`, `TMPDIR`, and cache
directories. CMake CUDA discovery is disabled. Every configure/install stage
is scanned for `/opt/rocm`, system CUDA, Triton LLVM, Miniforge, and old
temporary overlays. The final manifest is parsed and its source identities,
component state, local OpenCL files, managed gem5 input, and SHA-256 values are
verified.

## Activate for one shell

Use a disposable shell so activation is trivially reversible:

```bash
bash --noprofile --norc
rocm_prefix=$(scripts/setup_rocm_env.sh --print-prefix)
source "$rocm_prefix/activate"
```

The activation file sets `ROCM_SIM_ROOT`, `ROCM_PATH`, `HIP_PATH`, `HSA_PATH`,
`PATH`, `LD_LIBRARY_PATH`, `CMAKE_PREFIX_PATH`, and `PKG_CONFIG_PATH` to the
local prefix. It does not write shell startup files or system linker
configuration. Exit the disposable shell to restore the caller's environment.

## Direct OpenCL product path

`--all` builds and installs these user-facing files inside the versioned
prefix:

```text
$ROCM_SIM_ROOT/include/CL/cl.h
$ROCM_SIM_ROOT/lib/libOpenCL.so.1
$ROCM_SIM_ROOT/bin/opencl-vecadd
$ROCM_SIM_ROOT/share/self-amdgpu-runtime/opencl/vecadd.cl
```

The setup finishes by running the installed executable under `env -i`. That
executable calls `clBuildProgram`, which invokes only the prefix's pinned Clang
and device libraries, then automatically starts the repository's managed gem5
daemon, negotiates the private socket, submits the kernel, waits, copies C back,
checks the CPU oracle, and shuts the daemon down. There is no manually started
endpoint in this path.

Run the installed product command from the repository root without activating
the prefix:

```bash
set -euo pipefail
rocm_prefix=$(scripts/setup_rocm_env.sh --print-prefix)
run_home=$(mktemp -d "$rocm_prefix/tmp/opencl-home.XXXXXX")
chmod 0700 "$run_home"
env -i HOME="$run_home" PATH=/usr/bin:/bin LC_ALL=C \
  "$rocm_prefix/bin/opencl-vecadd" \
  "$rocm_prefix/share/self-amdgpu-runtime/opencl/vecadd.cl"
```

The successful result is one JSON line with at least:

```json
{"source_compiled":true,"gem5_execution":true,"output_correct":true,"mismatch_count":0,"fallback_count":0,"status":0,"stage":"complete"}
```

The accepted CP27 compilation produces a 5,160-byte gfx950 code object with
SHA-256
`314ede16940432996c9fe190115408bf42744a8ab7d0036bf07b931e39c4cb19`.
gem5 independently locks that image's ELF, metadata, descriptor, resources,
instruction bytes, launch shape, CU trace, and output oracle. A hash-different
OpenCL program is rejected rather than silently falling back.

### Compile the ordinary host executable yourself

The installed example is produced by CMake, but the same host source can be
compiled with an ordinary C compiler. This command links only the local OpenCL
library and embeds its versioned prefix as the runpath:

```bash
set -euo pipefail
rocm_prefix=$(scripts/setup_rocm_env.sh --print-prefix)
user_build=$(mktemp -d "$rocm_prefix/tmp/opencl-user-build.XXXXXX")
env -i PATH=/usr/bin:/bin LC_ALL=C /usr/bin/cc \
  -std=c11 -O2 -Wall -Wextra -Werror \
  -I"$rocm_prefix/include" \
  projects/self-amdgpu-runtime/examples/opencl/vecadd_host.c \
  -L"$rocm_prefix/lib" \
  -Wl,-rpath,"$rocm_prefix/lib" -Wl,--no-as-needed -lOpenCL \
  -o "$user_build/opencl-vecadd"

run_home=$(mktemp -d "$rocm_prefix/tmp/opencl-home.XXXXXX")
chmod 0700 "$run_home"
env -i HOME="$run_home" PATH=/usr/bin:/bin LC_ALL=C \
  "$user_build/opencl-vecadd" \
  projects/self-amdgpu-runtime/examples/opencl/vecadd.cl
```

The accepted capture for this path took about `0.94-0.95s` wall time on the
recorded checkout, while gem5 reported about `0.07s` host time. These numbers
are diagnostic, not a simulator performance guarantee, and are small enough
that profiling is not a prerequisite to the next operator gate.

### Current boundary

This gate proves one direct `vecadd` dispatch per OpenCL context. The local API
is intentionally synchronous, 1D, explicit-local-size, buffer-only, and
event-free. A second kernel submit on the same simulator connection, arbitrary
OpenCL code objects, normal Triton Python, model operators, multi-token
inference, TP, and CCL are not yet accepted. The older `gpuReadWrite` endpoint
remains a regression test, not the user command.

## Isolation checks

```bash
test ! -e /opt/rocm
scripts/setup_rocm_env.sh --verify-only
env | grep -E '^(CUDA|ROCM|HIP|HSA|LLVM_SYSPATH|TRITON|LD_LIBRARY_PATH)=' || true
ldd "${ROCM_SIM_ROOT}/bin/sagr-handshake" | grep -Ei 'libhsa|libhsakmt|libamdhip64' && exit 1 || true
ldd "${ROCM_SIM_ROOT}/bin/opencl-vecadd" | grep -Ei 'system.*/libOpenCL|libhsa|libhsakmt|libamdhip64' && exit 1 || true
ls -ld /dev/kfd /dev/dri 2>/dev/null || true
```

Device-node presence is not itself a failure; the retained process and fallback
evidence must show that neither node was opened and that no real AMD device was
silently substituted for the simulator.
