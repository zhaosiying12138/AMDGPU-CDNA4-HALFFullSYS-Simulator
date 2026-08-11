# Repository-local Triton simulator environment

The CP-0028 source candidate extends the repository-local ROCm prefix design
with the pinned Triton checkout and the external `gemsim_amd` backend. It does
not install or modify a system ROCm, HIP, HSA, CUDA, OpenCL ICD, Python package,
or device configuration.

The accepted CP-0028 correctness evidence came from a clean isolated package,
ordinary installed-wheel Python, and managed gem5 runs. CP-0029 now accepts a
fresh schema-8 repository-local prefix after independent verify-only, OpenCL,
Triton, provenance, pollution, and active-isolation gates. The v6 attempt was
stopped after a baked `/opt/rocm` default was found, the clean v7 attempt failed
a queue mock test race, and the first v8 attempt failed parent-Git provenance;
all three are retained as NON-PASSING evidence and must not be activated.

Build the complete isolated prefix from the repository root after CP-0029. The command can
take a long time on the first run because it builds the pinned LLVM and Triton
sources; every install and Python package remains below `env/rocm/`:

```bash
scripts/setup_rocm_env.sh --all --jobs 24
```

The build uses the host `/usr/bin/python3.14` interpreter only as a pinned
bootstrap input. Its matching `libpython3.14-dev` Debian package is downloaded
by an exact HTTPS URL, size, package identity, and SHA-256, then unpacked as
data below the versioned prefix with `dpkg-deb -x`. The script never runs
`apt install` or `dpkg --install`. The normalized private header tree, venv
interpreter, CMake Python selection, and installed Triton artifacts are all
recorded in the prefix manifest; Python packages are installed only into the
private venv. The retained pre-commit evidence separately binds the built wheel
to those installed artifacts.

Once a fresh prefix passes the complete setup gate, verify and activate that
exact versioned prefix, then run the normal Python entry point:

```bash
prefix=$(scripts/setup_rocm_env.sh --print-prefix)
scripts/setup_rocm_env.sh --verify-only --prefix "$prefix"
source "$prefix/activate"
python examples/triton/vecadd_correctness.py
```

The backend compiles the normal Triton `add_kernel`, loads the resulting HSACO
through the repository-local runtime, starts gem5 implicitly, stages CPU tensor
storage into simulated allocations, waits synchronously, copies results back,
and checks the PyTorch CPU oracle. The accepted gate launches the same compiled
kernel twice with independent deterministic inputs in one Python process and
requires `launch_count=2`, `reuse=true`, zero mismatches, and zero maximum
absolute error for both launches. A manually supplied bridge endpoint,
`libamdhip64`, `/opt/rocm`, `/dev/kfd`, `/dev/dri`, and CUDA are not part
of the product route.

The pinned Triton sources share a small set of compiler-internal NVGPU and
NVIDIA-to-LLVM objects between their AMD instrumentation and NVIDIA lowering.
Those objects are retained to build upstream common compiler code correctly;
the local wheel still exposes only the `amd` and `gemsim_amd` backends. It does
not install the NVIDIA Python backend, configure CUDA tools, load a CUDA/HSA/HIP
production runtime, or permit NVIDIA execution as a fallback.

CP-0028 and CP-0029 remain deliberately bounded to contiguous float32 vector
add at 98,432 elements, `BLOCK_SIZE=1024`, 97 programs, and two same-process
launches. The upstream tutorial benchmark block, arbitrary kernels, persistent
cross-kernel device allocations, the model operator matrix, stable multi-token
inference, TP, and CCL require later evidence gates. CP-0030 starts with the
minimum BF16 SiluAndMul contracts for decode `[1,7168] -> [1,3584]` and masked
prefill `[7,7168] -> [7,3584]`.

Subsequent LLVM/Triton project builds use `ccache`, `/usr/bin/clang`, and
`/usr/bin/ld.lld` with `-j24`. Compatible completed build directories and
caches are reused; cleanup is limited to directories proven obsolete by the
active evidence. The CP-0029 prefix was already in flight before this policy
was requested and is therefore kept unchanged.
