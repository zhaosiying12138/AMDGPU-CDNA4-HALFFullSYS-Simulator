#!/usr/bin/env bash
# usage: run_lds_capsule.sh <gem5-binary> [timeout-seconds]
#
# Runs the group-segment capsule against one gem5 binary with exactly the
# environment scripts/run_engine_lane.sh uses, so the capsule exercises the
# same ROCr/HIP/model-lib binaries the model lanes do. The identity of every
# one of them is printed first; a run that loads something else is void.
#
# The failure this reproduces is a hang, not a wrong answer, so the run is
# bounded by a timeout and the simulator is reaped afterwards. Exit 124 means
# the hang is present.
set -u

ROOT=/home/zhaosiying/amdgpu-sim
cd "$ROOT"

gem5="${1:?usage: run_lds_capsule.sh <gem5-binary> [timeout-seconds]}"
limit="${2:-600}"

PREFIX="${ROOT}/env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d"
# shellcheck disable=SC1091
source "${PREFIX}/etc/conda/activate.d/amdgpu-sim-rocm-pytorch.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/fastcopy_mode.sh" fast

STALE="${ROOT}/env/rocm/product-v1-f76db762609b346cb83b920cc82cd2b734b75cd31b8562e6536ad81275fe17e1"
HEAD_PRODUCT="${ROOT}/env/rocm/product-v1-4d9d40454031c7345f25da81b6781995b09a3b10e4dd66026e019306fc7ee39b"
# The self-runtime build the lanes use. Overridable so the capsule can be run
# against a known-good runtime while the working tree's copy is mid-edit; the
# resolved path and its hash are printed below either way.
RUNTIME_BUILD="${CAPSULE_RUNTIME_DIR:-${ROOT}/projects/self-amdgpu-runtime/build/cp28-runtime-clang}"

export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH}" | sed "s|${STALE}|${HEAD_PRODUCT}|g")"
export HSA_PATH="${HEAD_PRODUCT}"
export ROCM_SIM_ROOT="${HEAD_PRODUCT}"
export HSA_MODEL_TOPOLOGY="${HEAD_PRODUCT}/share/self-amdgpu-runtime/hsakmt-topology"
export LD_LIBRARY_PATH="${RUNTIME_BUILD}:${LD_LIBRARY_PATH}"
# The activate script preloads the ROCr of a product that predates every
# fix under test. Rewrite it rather than assigning over it: assigning drops
# the product preload and the run silently tests a different ROCr.
export LD_PRELOAD="${ROOT}/build/rocr_logging_preload.so:$(echo "${LD_PRELOAD:-}" | sed "s|${STALE}|${HEAD_PRODUCT}|g")"
export LD_PRELOAD="${LD_PRELOAD%:}"
export HSA_MODEL_LIB="${RUNTIME_BUILD}/libself_amdgpu_hsakmt_model.so.1"
export HSA_NO_SCRATCH_RECLAIM=1

export SAGR_MANAGED_GEM5_CONFIG="${ROOT}/projects/gem5/configs/example/gemsim/host_dispatch.py"
export SAGR_MANAGED_REPO_ROOT="${ROOT}"
# Own run root so lane_ownership.sh can tell this capsule's simulator apart
# from any model lane's, and so cleanup can never touch a lane.
export SAGR_MANAGED_RUN_ROOT="${SAGR_MANAGED_RUN_ROOT:-/tmp/sagr-lane-ldscapsule}"
mkdir -p "$SAGR_MANAGED_RUN_ROOT"

# The managed session starts gem5 with a scrubbed environment -- PATH, HOME,
# TMPDIR, XDG_CACHE_HOME, LC_ALL and nothing else -- so a variable exported
# here never reaches the simulator. A wrapper script is the only way to hand
# it one, which is also how the lanes pass --functional-fast.
# CAPSULE_FORCE_LDS_BYTES exists purely to reproduce the historical 64 KiB
# compute unit; leave it unset for a normal run.
gem5_real="$(readlink -f "$gem5")"
if [[ -n ${CAPSULE_FORCE_LDS_BYTES:-} ]]; then
  wrapper="${SAGR_MANAGED_RUN_ROOT}/gem5.lds${CAPSULE_FORCE_LDS_BYTES}.wrap"
  printf '#!/bin/sh\nGEMSIM_LDS_BYTES_PER_CU=%s exec %s "$@"\n' \
    "$CAPSULE_FORCE_LDS_BYTES" "$gem5_real" > "$wrapper"
  chmod +x "$wrapper"
  echo "IDENTITY forced_lds_bytes=${CAPSULE_FORCE_LDS_BYTES} wrapper=${wrapper}"
  gem5="$wrapper"
fi
export SAGR_MANAGED_GEM5="$gem5"
unset SAGR_OPENCL_ENDPOINT SAGR_OPENCL_SOCKET SAGR_OPENCL_GEM5_EXTERNAL
unset CUDA_VISIBLE_DEVICES

echo "IDENTITY gem5=${gem5_real} sha256=$(sha256sum "${gem5_real}" | cut -d' ' -f1)"
echo "IDENTITY managed_gem5=${SAGR_MANAGED_GEM5}"
echo "IDENTITY rocr=$(sha256sum "${HEAD_PRODUCT}/lib/libhsa-runtime64.so.1.21.0" | cut -d' ' -f1)"
echo "IDENTITY model_lib=$(sha256sum "${HSA_MODEL_LIB}" | cut -d' ' -f1)"
echo "IDENTITY topology_lds_kb=$(awk '/^lds_size_in_kb/{print $2}' "${HSA_MODEL_TOPOLOGY}/nodes/1/properties")"
echo "IDENTITY run_root=${SAGR_MANAGED_RUN_ROOT} timeout_s=${limit}"
echo "IDENTITY cases=${CAPSULE_LDS_CASES:-32k,96k,160k} groups=${CAPSULE_LDS_GROUPS:-2}"

start=$(date +%s)
timeout --signal=KILL "$limit" python3 "${ROOT}/tools/lds_capsule/lds_capsule.py"
rc=$?
end=$(date +%s)
echo "capsule_exit=${rc} wall_s=$((end - start))"

# Reap only simulators started under this capsule's own run root. Never match
# on a command-line pattern: that has destroyed healthy lanes here twice.
for pid in $(ps -eo pid,args --no-headers | awk '/[g]em5\.opt/{print $1}'); do
  if tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null \
       | grep -qF -- "${SAGR_MANAGED_RUN_ROOT}/"; then
    echo "reaping capsule gem5 pid=${pid}"
    kill -TERM "$pid" 2>/dev/null
  fi
done

exit "$rc"
