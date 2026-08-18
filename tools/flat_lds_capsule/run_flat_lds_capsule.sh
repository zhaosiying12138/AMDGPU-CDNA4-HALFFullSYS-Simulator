#!/usr/bin/env bash
# usage: run_flat_lds_capsule.sh <gem5-binary> [timeout-seconds]
#
# Runs the gfx950 global-to-LDS DMA capsule against one gem5 binary with the
# same environment scripts/run_engine_lane.sh gives a model lane, so it loads
# exactly the ROCr / HIP / model-lib the lanes load. Every one of those
# identities is printed before anything runs; a run that loaded something else
# is void as evidence in either direction.
#
# The capsule must execute inside a mount namespace with the host's NVML
# stub bound over /usr/lib/wsl/lib/libnvidia-ml.so.1, or hsa_init() returns
# 4104 and the process dies in about a second.
set -u

ROOT=/home/zhaosiying/amdgpu-sim
cd "$ROOT"

gem5="${1:?usage: run_flat_lds_capsule.sh <gem5-binary> [timeout-seconds]}"
limit="${2:-1800}"

PREFIX="${ROOT}/env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d"
# shellcheck disable=SC1091
source "${PREFIX}/etc/conda/activate.d/amdgpu-sim-rocm-pytorch.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/fastcopy_mode.sh" fast

STALE="${ROOT}/env/rocm/product-v1-f76db762609b346cb83b920cc82cd2b734b75cd31b8562e6536ad81275fe17e1"
HEAD_PRODUCT="${ROOT}/env/rocm/product-v1-4d9d40454031c7345f25da81b6781995b09a3b10e4dd66026e019306fc7ee39b"
RUNTIME_BUILD="${CAPSULE_RUNTIME_DIR:-${ROOT}/projects/self-amdgpu-runtime/build/cp28-runtime-clang}"

export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH}" | sed "s|${STALE}|${HEAD_PRODUCT}|g")"
export HSA_PATH="${HEAD_PRODUCT}"
export ROCM_SIM_ROOT="${HEAD_PRODUCT}"
export HSA_MODEL_TOPOLOGY="${HEAD_PRODUCT}/share/self-amdgpu-runtime/hsakmt-topology"
export LD_LIBRARY_PATH="${RUNTIME_BUILD}:${LD_LIBRARY_PATH}"
# Prepend, never assign: assigning drops the product ROCr preload and the run
# silently tests a ROCr that predates every fix under test.
export LD_PRELOAD="$(echo "${LD_PRELOAD:-}" | sed "s|${STALE}|${HEAD_PRODUCT}|g")"
export LD_PRELOAD="${LD_PRELOAD%:}"
export HSA_MODEL_LIB="${RUNTIME_BUILD}/libself_amdgpu_hsakmt_model.so.1"
export HSA_NO_SCRATCH_RECLAIM=1

export SAGR_MANAGED_GEM5_CONFIG="${ROOT}/projects/gem5/configs/example/gemsim/host_dispatch.py"
export SAGR_MANAGED_REPO_ROOT="${ROOT}"
# Own run root, so lane_ownership.sh can never confuse this capsule's
# simulator with a model lane's and cleanup can never touch a lane.
export SAGR_MANAGED_RUN_ROOT="${SAGR_MANAGED_RUN_ROOT:-/tmp/sagr-lane-ldsdma}"
mkdir -p "$SAGR_MANAGED_RUN_ROOT"

gem5_real="$(readlink -f "$gem5")"
export SAGR_MANAGED_GEM5="$gem5_real"
unset SAGR_OPENCL_ENDPOINT SAGR_OPENCL_SOCKET SAGR_OPENCL_GEM5_EXTERNAL
unset CUDA_VISIBLE_DEVICES

echo "IDENTITY repo_head=$(git -C "${ROOT}" rev-parse HEAD)"
echo "IDENTITY gem5_head=$(git -C "${ROOT}/projects/gem5" rev-parse HEAD)"
echo "IDENTITY gem5=${gem5_real}"
echo "IDENTITY gem5_sha256=$(sha256sum "${gem5_real}" | cut -d' ' -f1)"
echo "IDENTITY gem5_mtime=$(date -r "${gem5_real}" --iso-8601=seconds)"
echo "IDENTITY rocr_sha256=$(sha256sum "${HEAD_PRODUCT}/lib/libhsa-runtime64.so.1.21.0" | cut -d' ' -f1)"
echo "IDENTITY model_lib_sha256=$(sha256sum "${HSA_MODEL_LIB}" | cut -d' ' -f1)"
echo "IDENTITY probe_sha256=$(sha256sum "${ROOT}/tools/flat_lds_capsule/lds_dma_probe.hip" | cut -d' ' -f1)"
echo "IDENTITY run_root=${SAGR_MANAGED_RUN_ROOT} timeout_s=${limit}"
echo "IDENTITY cases=${CAPSULE_LDS_DMA_CASES:-all} groups=${CAPSULE_LDS_DMA_GROUPS:-2}"

: > /tmp/empty-nvml.so
start=$(date +%s)
unshare -r -m bash -c '
  mount --bind /tmp/empty-nvml.so /usr/lib/wsl/lib/libnvidia-ml.so.1
  exec timeout --signal=KILL "$0" python3 "$1"
' "$limit" "${ROOT}/tools/flat_lds_capsule/flat_lds_capsule.py" &
capsule_pid=$!

# A dead simulator leaves the host blocked in hipStreamSynchronize until the
# timeout expires, which turns a two-second answer into a fifteen-minute one.
# Watch the simulator's own log and stop the capsule as soon as it aborts.
(
  while kill -0 "$capsule_pid" 2>/dev/null; do
    hit=$(grep -m1 -hE "^[^ ]*: (fatal|panic):" \
            "${SAGR_MANAGED_RUN_ROOT}"/*/gem5.log 2>/dev/null || true)
    if [[ -n $hit ]]; then
      echo "SIMULATOR_FATAL ${hit}"
      kill -KILL "$capsule_pid" 2>/dev/null
      exit 0
    fi
    sleep 2
  done
) &
watchdog=$!

wait "$capsule_pid"
rc=$?
kill -TERM "$watchdog" 2>/dev/null
wait "$watchdog" 2>/dev/null
echo "capsule_exit=${rc} wall_s=$(( $(date +%s) - start ))"

for log in "${SAGR_MANAGED_RUN_ROOT}"/*/gem5.log; do
  [[ -r $log ]] || continue
  grep -hE "^[^ ]*: (fatal|panic):" "$log" | while read -r line; do
    echo "SIMULATOR_ERROR ${log}: ${line}"
  done
done

# Reap only simulators started under this capsule's own run root. Never match
# on a command-line substring: that has destroyed healthy lanes here twice.
for pid in $(ps -eo pid,args --no-headers | awk '/[g]em5\.opt/{print $1}'); do
  [[ -r "/proc/$pid/cmdline" ]] || continue
  if tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null \
       | grep -qF -- "${SAGR_MANAGED_RUN_ROOT}/"; then
    echo "reaping capsule gem5 pid=${pid}"
    kill -TERM "$pid" 2>/dev/null
  fi
done

exit "$rc"
