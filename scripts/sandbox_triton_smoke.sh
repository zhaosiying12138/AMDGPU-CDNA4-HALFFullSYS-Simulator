#!/usr/bin/env sh
# Run the upstream Triton AMD softmax smoke inside an AgentENV sandbox.
#
#   aenv exec <sandbox-id> -- /bin/sh /home/zhaosiying/amdgpu-sim/scripts/sandbox_triton_smoke.sh
#
# The sandbox is a Fedora 42 microVM with no /dev/kfd, no /dev/dri and no
# /usr/lib/wsl, so there is nothing to mask and nothing to fall back to: if a
# kernel produces a number, the simulated device produced it.
#
# This is the in-sandbox peer of scripts/run_engine_lane.sh and deliberately
# shares its two hard-won environment rules:
#
#   * only the freshly built self-runtime is prepended to LD_LIBRARY_PATH, and
#   * the product's lib/ stays *after* the stock ROCm 7.2.3 sysroot, because
#     PyTorch's precompiled device code was built against 7.2.3 and loading the
#     HEAD product's HIP invalidates every one of those code objects
#     (hipErrorInvalidImage at the first module load).
#
# ROCr still resolves to the product: it is preloaded explicitly below and no
# other libhsa-runtime64.so.1 exists under the prefix.
set -u

R=${SANDBOX_REPO_ROOT:-/home/zhaosiying/amdgpu-sim}
P=$R/env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d
PROD=$R/env/rocm/product-v1-4d9d40454031c7345f25da81b6781995b09a3b10e4dd66026e019306fc7ee39b
RT=$R/projects/self-amdgpu-runtime/build/cp28-runtime-clang
SYSROOT=$P/rocm-sysroot/opt/rocm-7.2.3

export LD_LIBRARY_PATH=$RT:$P/system-runtime/usr/lib/x86_64-linux-gnu:$SYSROOT/lib:$SYSROOT/lib/rocm_sysdeps/lib:$P/lib:$PROD/lib:$P/rocm-sysroot/usr/lib/x86_64-linux-gnu:$P/rocm-sysroot/lib/x86_64-linux-gnu
# Prepended, never assigned over: an assignment silently displaces a diagnostic
# preload the caller set, and the reverse mistake has voided a run before.
export LD_PRELOAD=$PROD/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}

export ROCM_PATH=$SYSROOT
export HIP_PATH=$SYSROOT
export HIP_PLATFORM=amd
export HSA_PATH=$PROD
export ROCM_SIM_ROOT=$PROD
export HSA_MODEL_LIB=$RT/libself_amdgpu_hsakmt_model.so.1
export HSA_MODEL_TOPOLOGY=$PROD/share/self-amdgpu-runtime/hsakmt-topology
export HSA_ENABLE_DXG_DETECTION=0
export HSA_ENABLE_INTERRUPT=0
export PYTORCH_ROCM_ARCH=gfx950

# Generic fast copy, both gates.
export HSA_ENABLE_DTIF_FAST_COPY=1
export SAGR_HSAKMT_MODEL_FAST_COPY=1

# gem5.opt is built on the host against a newer glibc than this guest image
# ships, so it is launched through the host loader shim staged at
# /usr/local/bin/gem5-hostloader rather than rebuilt inside the sandbox.
export SAGR_MANAGED_GEM5=${SAGR_MANAGED_GEM5:-/usr/local/bin/gem5-hostloader}
export SAGR_MANAGED_GEM5_CONFIG=$R/projects/gem5/configs/example/gemsim/host_dispatch.py
export SAGR_MANAGED_REPO_ROOT=$R
# One run root per workload keeps this smoke's simulator distinguishable from
# anything else in the sandbox; see scripts/lane_ownership.sh.
export SAGR_MANAGED_RUN_ROOT=${SAGR_MANAGED_RUN_ROOT:-/tmp/sagr-sandbox-triton}
mkdir -p "$SAGR_MANAGED_RUN_ROOT"

# Unchanged upstream Triton: the in-tree AMD backend, not the project's
# gemsim_hip subclass. Triton compiles a small launcher .so at run time and
# needs a host C compiler for it.
unset TRITON_DEFAULT_BACKEND
export CC=${CC:-/usr/bin/gcc}
export CXX=${CXX:-/usr/bin/g++}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/triton-cache}
export PYTHONNOUSERSITE=1
mkdir -p "$TRITON_CACHE_DIR"

# --- identity ---------------------------------------------------------------
# A run whose loaded binaries do not descend from the commit under test is void
# as evidence, so record them before anything executes.
echo "# in-sandbox identity"
echo "repo_root=$R"
echo "managed_gem5=$SAGR_MANAGED_GEM5"
echo "managed_gem5_sha256=$(sha256sum "$SAGR_MANAGED_GEM5" | cut -d' ' -f1)"
# SAGR_MANAGED_GEM5 is usually the host-loader shim rather than the simulator,
# so hashing it alone would happily certify the wrong gem5. Record the ELF the
# shim actually execs. Reading a fixed prefix keeps this safe if it is itself a
# binary.
gem5_elf=$(head -c 4096 "$SAGR_MANAGED_GEM5" 2>/dev/null \
           | grep -a -o -- '/[^ "]*gem5[^ "]*\.opt' | tail -1)
if [ -n "${gem5_elf:-}" ] && [ -r "$gem5_elf" ]; then
  echo "gem5_elf=$gem5_elf"
  echo "gem5_elf_sha256=$(sha256sum "$gem5_elf" | cut -d' ' -f1)"
fi
echo "rocr_sha256=$(sha256sum "$PROD/lib/libhsa-runtime64.so.1.21.0" | cut -d' ' -f1)"
echo "model_lib_sha256=$(sha256sum "$HSA_MODEL_LIB" | cut -d' ' -f1)"
echo "runtime_sha256=$(sha256sum "$RT/libself_amdgpu_runtime.so.0.8.0" | cut -d' ' -f1)"
echo "fastcopy=$HSA_ENABLE_DTIF_FAST_COPY/$SAGR_HSAKMT_MODEL_FAST_COPY"
echo "kfd_present=$([ -e /dev/kfd ] && echo yes || echo no)"
echo "dri_present=$([ -e /dev/dri ] && echo yes || echo no)"
echo "wsl_present=$([ -d /usr/lib/wsl ] && echo yes || echo no)"
echo "started=$(date -Is)"

# A marker whose mtime bounds this run, so the evidence step below can tell
# this smoke's simulator session apart from any earlier one in the sandbox.
marker=$(mktemp)
trap 'rm -f "$marker"' EXIT

cd "$R"
"$P/bin/python3.12" examples/quickstart/triton_softmax.py "$@"
status=$?

# --- evidence ---------------------------------------------------------------
# Correct numbers alone do not prove the simulated device produced them, so
# report the dispatch trace this run actually retired. Both roots are searched
# because a self-runtime predating SAGR_MANAGED_RUN_ROOT still uses TMPDIR.
retired=0
sessions=0
traces=$(find "$SAGR_MANAGED_RUN_ROOT" "${TMPDIR:-/tmp}" -maxdepth 2 \
         -name dispatch-trace.jsonl -newer "$marker" 2>/dev/null)
for trace in $traces; do
  echo "dispatch_trace=$trace"
  retired=$((retired + $(grep -c native_execution_retired "$trace" 2>/dev/null || echo 0)))
  sessions=$((sessions + $(grep -c native_execution_session_complete "$trace" 2>/dev/null || echo 0)))
done
echo "retired_dispatches=$retired"
echo "session_complete_records=$sessions"
if [ "$status" -eq 0 ] && [ "$retired" -eq 0 ]; then
  echo "no dispatch trace retired a kernel; the result is not device evidence" >&2
  status=3
fi
echo "finished=$(date -Is) status=$status"
exit "$status"
