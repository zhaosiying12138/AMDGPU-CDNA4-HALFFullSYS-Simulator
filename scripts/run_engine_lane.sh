#!/usr/bin/env bash
# One runner for every acceptance lane: {SGLang, vLLM} x {TP1, TP2, TP16}.
#
#   scripts/run_engine_lane.sh --engine sglang --tp 1 <logfile>
#   scripts/run_engine_lane.sh --engine vllm   --tp 16 <logfile>
#   scripts/run_engine_lane.sh --engine sglang --tp 1 --product-hip <logfile>
#
# This replaces six near-identical scripts. They had drifted apart in ways that
# cost real time: only one of them prepended the freshly built runtime to
# LD_LIBRARY_PATH, so the other silently ran a stale library, and only one
# pinned the HEAD product, so the other loaded a ROCm build predating every
# fast-copy commit. A single runner makes such a divergence impossible to
# introduce by accident, and the identity header below makes it impossible to
# miss if it happens anyway.
#
# Both engines are stock upstream. Nothing here registers a project operator,
# replaces a model or layer, or patches engine source; the only adaptation is
# to the self-runtime interface, which is the thing under test.
set -u

ROOT=/home/zhaosiying/zcode-lane
# zcode worktree isolation: the immutable conda product, model weights and
# frozen goldens are shared with the primary tree read-only, but mutable caches
# must stay private.  The conda activate script below points TRITON_CACHE_DIR
# at a shared conda-state dir; two lanes writing the same cache key
# concurrently can corrupt compilation, so repoint it under this worktree
# before anything imports triton.  SAGR_LANE_CACHE_ROOT exists for the
# perf-ablation harness: pointing it at a fresh directory gives a cold Triton
# cache without touching the persistent warm one.
ZCODE_CACHE="${SAGR_LANE_CACHE_ROOT:-${ROOT}/artifacts/zcode-cache}"
mkdir -p "${ZCODE_CACHE}/triton" "${ZCODE_CACHE}/xdg"
cd "$ROOT"

engine=""
tp=""
model=""
dummy_weights=0
capsule=""
max_new_tokens=1
product_hip=0
debug_layer_gate=0
# Number of compute units to expose in the gem5 host bridge.  The managed
# runtime deliberately scrubs arbitrary environment variables before spawning
# gem5, so a lane-owned wrapper is generated below when this is set.  Keeping
# the knob here makes a multi-CU model run reproducible instead of dependent on
# an out-of-tree temporary wrapper.
compute_units="${GEMSIM_NUM_COMPUTE_UNITS:-}"
# Which simulator to run. The optimised build plus --functional-fast is
# ~3x faster on the representative kernel and was shown to produce
# byte-identical results and an identical retired-dispatch sequence, but
# the accurate binary stays the default so a lane opts in explicitly.
gem5="${ROOT}/projects/gem5/build/VEGA_X86/gem5.opt"

while (($#)); do
  case "$1" in
    --engine) engine=${2:?}; shift 2 ;;
    --tp) tp=${2:?}; shift 2 ;;
    --model) model=${2:?}; shift 2 ;;
    --dummy-weights) dummy_weights=1; shift ;;
    --max-new-tokens) max_new_tokens=${2:?}; shift 2 ;;
    --product-hip) product_hip=1; shift ;;
    --debug-layer-gate) debug_layer_gate=1; shift ;;
    --gem5) gem5=${2:?}; shift 2 ;;
    --compute-units) compute_units=${2:?}; shift 2 ;;
    --fast) gem5="${ROOT}/projects/gem5/build/VEGA_X86/gem5.opt.fastwrap"; shift ;;
    # Run a standalone capsule instead of an engine, under this script's exact
    # environment and NVML isolation. A capsule that rebuilds the environment
    # for itself drifts from the lanes and then proves nothing about them --
    # and outside the isolation it dies immediately with hsa_init 4104, which
    # has already cost several confused debugging attempts.
    --capsule) capsule=${2:?}; shift 2 ;;
    *) break ;;
  esac
done

log="${1:?usage: run_engine_lane.sh --engine E --tp N [--model M] [--dummy-weights] <logfile>}"

if [[ -z $capsule ]]; then
  case "$engine" in
    sglang|vllm) ;;
    *) printf 'engine must be sglang or vllm\n' >&2; exit 2 ;;
  esac
elif [[ ! -r $capsule ]]; then
  printf 'capsule is not readable: %s\n' "$capsule" >&2; exit 2
fi
if ! [[ $tp =~ ^[0-9]+$ ]] || (( tp < 1 )); then
  printf 'tp must be a positive integer\n' >&2; exit 2
fi
if ! [[ $max_new_tokens =~ ^[0-9]+$ ]] || (( max_new_tokens < 1 )); then
  printf 'max new tokens must be a positive integer\n' >&2
  exit 2
fi
if (( debug_layer_gate )) && { [[ $engine != sglang ]] || (( tp != 1 )); }; then
  printf '%s\n' '--debug-layer-gate requires --engine sglang --tp 1' >&2
  exit 2
fi
if (( debug_layer_gate && dummy_weights )); then
  printf '%s\n' '--debug-layer-gate requires checkpoint weights' >&2
  exit 2
fi
if [[ -n $compute_units ]] && {
  ! [[ $compute_units =~ ^[0-9]+$ ]] || (( compute_units < 1 ));
}; then
  printf 'compute units must be a positive integer\n' >&2
  exit 2
fi

# The ladder pairs a model with a degree of parallelism: 0.8B carries TP1 and
# TP2, 9B carries TP16.
if [[ -z $model ]]; then
  if (( tp >= 16 )); then model="${ROOT}/models/Qwen3.5-9B"; else model="${ROOT}/models/Qwen3.5-0.8B"; fi
fi
if [[ ! -d $model ]]; then
  printf 'model directory is absent: %s\n' "$model" >&2
  exit 2
fi

# --- environment ------------------------------------------------------------
PREFIX="${ROOT}/env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d"
# shellcheck disable=SC1091
source "${PREFIX}/etc/conda/activate.d/amdgpu-sim-rocm-pytorch.sh"
# Override the activate script's shared conda-state cache with the private
# worktree cache prepared above (see the ROOT block comment for why).
# ZCODE_SHARE_CACHES=1 is a diagnostic A/B switch that keeps the lane on the
# shared caches exactly like the primary tree's lanes, to attribute an env
# failure to the cache split instead of the code under test.
if [[ -z ${ZCODE_SHARE_CACHES:-} ]]; then
  export TRITON_CACHE_DIR="${ZCODE_CACHE}/triton"
  export XDG_CACHE_HOME="${ZCODE_CACHE}/xdg"
fi
# SAGR_LANE_FASTCOPY_MODE is the perf-ablation override for the DTIF
# fast-copy gate: "fast" (default, the accepted lanes) or "legacy" (every
# H2D/D2H copy becomes a simulated blit-kernel dispatch).
# shellcheck disable=SC1091
source "${ROOT}/scripts/fastcopy_mode.sh" "${SAGR_LANE_FASTCOPY_MODE:-fast}"

# The activate script of this prefix exports the product that predates every
# fast-copy commit. Rewriting the paths is what makes the run test the code
# that is actually under review -- a run that skipped this step once spent 70
# minutes proving nothing.
STALE="${ROOT}/env/rocm/product-v1-f76db762609b346cb83b920cc82cd2b734b75cd31b8562e6536ad81275fe17e1"
HEAD_PRODUCT="${ROOT}/env/rocm/product-v1-4d9d40454031c7345f25da81b6781995b09a3b10e4dd66026e019306fc7ee39b"
RUNTIME_BUILD="${SAGR_RUNTIME_LIBRARY_DIR:-${ROOT}/projects/self-amdgpu-runtime/build/cp28-runtime-clang}"
# SAGR_RUNTIME_LIBRARY_DIR (optional): point the lane at a rebuilt runtime
# tree (it must contain libself_amdgpu_hsakmt_model.so.1 and
# libself_amdgpu_runtime.so.1). Used by the TP16 lane to run the build whose
# SMI registry holds 64 device slots instead of 16: a TP16 engine holds 18
# concurrent registry leases (16 schedulers + engine process + detokenizer),
# and with 16 slots the 17th/18th hsa_init failed OUT_OF_RESOURCES after its
# gem5 session had already handshaken, killing a random scheduler with
# "No CUDA GPUs are available".
# Keep the product as the base, but default this worktree to the rebuilt ROCr
# stage: the conda product prefix named by the activate script predates every
# fast-copy commit in projects/rocm-systems, and a lane that loads it falls
# back from DTIF fast copy to blit-kernel copies and dies at the first device
# allocation with hipErrorInvalidImage.  SAGR_ROCR_LIBRARY_DIR still overrides
# (the upstream switch); only the default changes.  model DSO, topology,
# rocminfo, and HIP remain owned by HEAD_PRODUCT.
ROCR_LIBRARY_DIR="${SAGR_ROCR_LIBRARY_DIR:-${ROOT}/build/rocr-stage-0401e8cd/lib}"
if [[ ! -f "${ROCR_LIBRARY_DIR}/libhsa-runtime64.so.1" ]]; then
  printf 'ROCr library directory is missing libhsa-runtime64.so.1: %s\n' \
    "$ROCR_LIBRARY_DIR" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH}" | sed "s|${STALE}|${HEAD_PRODUCT}|g")"
export HSA_PATH="${HEAD_PRODUCT}"
export ROCM_SIM_ROOT="${HEAD_PRODUCT}"
# The topology declares how many GPUs the simulated stack exposes, and the
# product ships a single-GPU one. A tensor-parallel engine calls
# set_device(rank) on every rank, so TP2 dies immediately with
#
#   torch.AcceleratorError: CUDA error: invalid device ordinal
#
# unless the topology has that many GPU nodes. Topologies are generated by
# projects/self-amdgpu-runtime/tools/hsakmt-model-topology.py; the model
# supports up to SAGR_HSAKMT_MODEL_MAXIMUM_VISIBLE_GPUS = 16, which is exactly
# what the TP16 rung needs.
if (( tp > 1 )); then
  export HSA_MODEL_TOPOLOGY="${ROOT}/artifacts/topology/gpu-${tp}"
  if [[ ! -d $HSA_MODEL_TOPOLOGY ]]; then
    python3 "${ROOT}/projects/self-amdgpu-runtime/tools/hsakmt-model-topology.py" \
      --output-dir "$HSA_MODEL_TOPOLOGY" --gpu-count "$tp" >/dev/null || {
        printf 'could not generate a %s-GPU topology\n' "$tp" >&2; exit 2; }
  fi
  # The generated topology advertises a full gfx950 (simd_count 1024 =
  # 256 compute units), but the simulator instantiates NUM_COMPUTE_UNITS
  # (default 4). That mismatch is not just cosmetic: engine libraries size
  # kernels by the advertised CU count (aiter's get_cu_num() parses this
  # very topology through the product rocminfo shim), and its split-K GEMMs
  # assign early workgroups to poll workspace flags that later workgroups
  # publish -- a schedule that assumes near-full-grid co-residency. With a
  # 4-CU simulator and in-order workgroup admission, the first resident
  # cohort is all pollers, the publishers are never admitted, and both TP
  # ranks livelock deterministically (observed at dispatch 1618/1619 on
  # every SGLang TP2 run). Publish exactly what is simulated:
  # simd_count = NUM_COMPUTE_UNITS * simd_per_cu.
  gemsim_cus="${GEMSIM_NUM_COMPUTE_UNITS:-4}"
  if [[ ! $gemsim_cus =~ ^[0-9]+$ ]] || (( gemsim_cus < 1 )); then
    gemsim_cus=4
  fi
  gemsim_simds=$(( gemsim_cus * 4 ))
  for node_props in "${HSA_MODEL_TOPOLOGY}"/nodes/*/properties; do
    [[ -f $node_props ]] || continue
    # Only GPU nodes carry a non-zero simd_count; the CPU node's 0 marks it
    # as a CPU, and rewriting it would publish a bogus gfx000 GPU agent.
    current_simds=$(sed -n 's/^simd_count \([0-9][0-9]*\)$/\1/p' "$node_props")
    if [[ -n $current_simds ]] && (( current_simds > 0 )) && \
       (( current_simds != gemsim_simds )); then
      sed -i "s/^simd_count [0-9][0-9]*$/simd_count ${gemsim_simds}/" \
        "$node_props"
      printf 'topology %s: simd_count -> %d (%d CUs) to match the simulator\n' \
        "$node_props" "$gemsim_simds" "$gemsim_cus" >&2
    fi
  done
else
  export HSA_MODEL_TOPOLOGY="${HEAD_PRODUCT}/share/self-amdgpu-runtime/hsakmt-topology"
fi

# Order matters. The default keeps the previously accepted stock ROCm 7.2.3
# HIP plus HEAD-product ROCr combination. `--product-hip` is an explicit,
# auditable comparison mode: the immutable product lib directory comes first,
# so both libamdhip64 and its matching libamd_comgr resolve from the product.
# HIP_PATH/ROCM_PATH remain the pinned compiler environment; this switch only
# changes the runtime libraries loaded by the model process.
if (( product_hip )); then
  hip_mode=product
  # Keep the mutable runtime build first: the product also contains an older
  # copy of libself_amdgpu_runtime.so.1, and letting that copy win silently
  # disables run-root confinement. The build has no HIP soname, so product HIP
  # and COMGR still win over the Conda copies at the next path element.
  export LD_LIBRARY_PATH="${RUNTIME_BUILD}:${HEAD_PRODUCT}/lib:${LD_LIBRARY_PATH}"
  # The generic HIP capsule accepts this explicit runtime path and records it
  # in its result. Frameworks still bind through the normal ELF loader; the
  # variable is a lane identity aid, not a framework-specific hook.
  export HIP_RUNTIME_LIBRARY="${HEAD_PRODUCT}/lib/libamdhip64.so.7"
else
  hip_mode=stock
  export LD_LIBRARY_PATH="${RUNTIME_BUILD}:${LD_LIBRARY_PATH}"
fi
# The selected ROCr is first and explicit.  Do not retain the activation
# script's stale preload when a lane requests an isolated rebuilt stage.
export LD_LIBRARY_PATH="${RUNTIME_BUILD}:${ROCR_LIBRARY_DIR}:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${ROOT}/build/rocr_logging_preload.so:${ROCR_LIBRARY_DIR}/libhsa-runtime64.so.1"
export HSA_MODEL_LIB="${RUNTIME_BUILD}/libself_amdgpu_hsakmt_model.so.1"

# Device discovery belongs to this product, and some consumers reach it by
# running a *tool* rather than by calling a library: aiter resolves the live
# gfx arch with shutil.which("rocminfo") and parses its output, and documents
# that get_gfx_runtime() ignores GPU_ARCHS, so no environment variable can
# stand in. Upstream rocminfo cannot answer here -- it needs the amdgpu kernel
# module, and on this host it prints "hsa_init Failed" with no gfx line at all,
# which made aiter deduce an arch of [''] and abort a lane after 1447 retired
# dispatches. Put the product's own rocminfo first on PATH. The product prefix
# is immutable and the upstream sysroot is not ours to overwrite, so the shim
# lives in a directory this runner owns and regenerates.
TOOL_SHIM="${ROOT}/artifacts/tool-shim"
mkdir -p "$TOOL_SHIM"
ln -sfn "${RUNTIME_BUILD}/sagr-rocminfo" "${TOOL_SHIM}/rocminfo"
ln -sfn "${ROOT}/tools/sim_amdgpu_arch.sh" "${TOOL_SHIM}/amdgpu-arch"
export SAGR_SIM_ROCMINFO="${TOOL_SHIM}/rocminfo"
export PATH="${TOOL_SHIM}:${PATH}"

# AITER's aiter_meta helper evaluates amdgpu-arch at module import time. The
# upstream utility returns success with no output when no real KMD is present,
# which turns its DEFAULT_GPU_ARCH (and later GPU_ARCHS) into an empty string.
# Resolve an empty/unset value from the same topology-backed product tool that
# get_gfx_runtime() uses. This is a temporary bring-up adapter and remains
# architecture-generic: the shim reports every GPU name the topology exposes.
gpu_archs_value=${GPU_ARCHS:-}
if [[ -z "${gpu_archs_value//[[:space:]]/}" ]]; then
  GPU_ARCHS="$(${TOOL_SHIM}/amdgpu-arch | tr '\n' ';' | sed 's/;$//')"
  if [[ -z "$GPU_ARCHS" ]]; then
    printf 'could not derive GPU_ARCHS from simulator device discovery\n' >&2
    exit 2
  fi
  export GPU_ARCHS
fi

export SAGR_MANAGED_GEM5="${SAGR_MANAGED_GEM5:-$gem5}"
# The managed config path is normally this worktree's script; an already-set
# value lets a diagnostic lane point at another worktree's config (e.g. the
# hybrid-CTA branch whose script carries extra options the main tree lacks).
export SAGR_MANAGED_GEM5_CONFIG="${SAGR_MANAGED_GEM5_CONFIG:-${ROOT}/projects/gem5/configs/example/gemsim/host_dispatch.py}"
export SAGR_MANAGED_REPO_ROOT="${ROOT}"
# SAGR_MANAGED_RUN_ROOT is inherited from the lane supervisor. It confines this
# lane's simulators to their own directory so progress accounting and cleanup
# can tell them apart from another lane's; see scripts/lane_ownership.sh.

# managed_session.c launches gem5 with a seven-entry environment and therefore
# cannot pass GEMSIM_NUM_COMPUTE_UNITS through as a normal export.  Generate a
# private executable in the lane root that restores exactly this one simulator
# setting before replacing itself with the selected gem5 binary.  The wrapper
# is intentionally lane-owned and included in the identity record below.
if [[ -n $compute_units ]]; then
  wrapper_root="${SAGR_MANAGED_RUN_ROOT:-${ROOT}/artifacts/lanes}"
  mkdir -p "$wrapper_root"
  gem5_base="$gem5"
  gem5_wrapper="${wrapper_root}/gem5-cu${compute_units}.sh"
  {
    printf '%s\n' '#!/bin/sh'
    printf 'export GEMSIM_NUM_COMPUTE_UNITS=%q\n' "$compute_units"
    printf 'exec %q "$@"\n' "$gem5_base"
  } >"$gem5_wrapper"
  chmod 700 "$gem5_wrapper"
  gem5="$gem5_wrapper"
  export SAGR_MANAGED_GEM5="${SAGR_MANAGED_GEM5:-$gem5}"
fi

export TRITON_DEFAULT_BACKEND=gemsim_hip
export TRITON_BACKENDS_IN_TREE=0
# SAGR_LANE_AUTOTUNE_MODE is the perf-ablation override: "correctness"
# (default, one dispatch per autotune candidate) or "device" (full do_bench
# scan on the simulator -- the unoptimized path).
export GEMSIM_HIP_AUTOTUNE_MODE="${SAGR_LANE_AUTOTUNE_MODE:-correctness}"
export TRITON_CACHE_AUTOTUNING=1
# ROCr refuses to run multi-agent collectives without this and aborts with
#   [FATAL ERROR]: HSA_NO_SCRATCH_RECLAIM=1 must be set
# It is an upstream-documented ROCm setting, not a project workaround.
export HSA_NO_SCRATCH_RECLAIM=1
unset SAGR_OPENCL_ENDPOINT SAGR_OPENCL_SOCKET SAGR_OPENCL_GEM5_EXTERNAL
unset CUDA_VISIBLE_DEVICES

if [[ $engine == sglang ]]; then
  if (( debug_layer_gate )); then
    layer_gate_root="${ROOT}/tools/qwen35_sglang_layer_gate"
    layer_gate_output="$(dirname "$log")/layer-gate"
    export SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT="$layer_gate_output"
    export SAGR_QWEN35_SGLANG_LAYER_GATE_GOLDEN="${ROOT}/artifacts/qwen35-nvidia-golden/20260812-prefill2-max24-v1"
    export SAGR_QWEN35_OPERATOR_GOLDEN="${ROOT}/artifacts/qwen35-nvidia-operator-golden/20260819-prefill2-layer0-v3"
    export SAGR_TRITON_LAUNCH_LOG="$(dirname "$log")/triton-launches.jsonl"
    export PYTHONPATH="${layer_gate_root}:${ROOT}/tools/triton_launch_probe:${ROOT}/projects/sglang-0.5.17:${ROOT}/env/sglang-overlay-cp312"
  else
    export PYTHONPATH="${ROOT}/projects/sglang-0.5.17:${ROOT}/env/sglang-overlay-cp312"
  fi
  # aiter's tuned-GEMM table has no valid tiling for the decode m=1 GEMM
  # shapes (n=8192 k=1024 rejected by selection_filter), which kills the
  # engine at the first decode step.  SAGR_SGLANG_USE_AITER=0 routes the
  # linear layers through the standard torch F.linear path while leaving the
  # attention backend selection untouched.
  export SGLANG_USE_AITER="${SAGR_SGLANG_USE_AITER:-1}"
  # aiter resolves its bf16 tuned-GEMM table from a machine-global file
  # (/tmp/aiter_configs/bf16_tuned_gemm.csv).  Two concurrent lanes once
  # rewrote that file mid-run: the victim lane then selected a different,
  # freshly JIT-compiled GEMM for shapes that earlier runs had served from
  # the table, and the fresh kernel chased a stale pointer into a gem5
  # host-native load panic (zcode-decode-18, capture fatal-20260821T122252;
  # the same signature explains the never-reproduced "Invalid tiling" crash
  # of lgate-4).  Pin the table per lane to a frozen copy so dispatch
  # decisions are lane-local and reproducible.
  aiter_cfg_dir="${SAGR_MANAGED_RUN_ROOT:-/tmp}/aiter-config"
  mkdir -p "$aiter_cfg_dir"
  if [[ ! -s ${aiter_cfg_dir}/bf16_tuned_gemm.csv ]]; then
    # Seed from the package-owned table (deterministic per pinned env), not
    # the mutable /tmp copy whose contents depend on what other lanes last
    # merged into it.  aiter's import merges the package model_configs into
    # this file at startup, so the lane-local copy ends up complete either
    # way, and only this lane ever writes it.
    for src in \
      "${PREFIX}/lib/python3.12/site-packages/aiter/configs/bf16_tuned_gemm.csv" \
      /tmp/aiter_configs/bf16_tuned_gemm.csv; do
      if [[ -r $src ]]; then
        cp "$src" "${aiter_cfg_dir}/bf16_tuned_gemm.csv"
        break
      fi
    done
  fi
  export AITER_CONFIG_GEMM_BF16="${aiter_cfg_dir}/bf16_tuned_gemm.csv"
  export FLA_CACHE_RESULTS=1
else
  # vLLM runs on the formal Triton path: the unchanged upstream AMD hip
  # compiler and driver, which is what GOAL.md designates and what leaves
  # gemsim_hip as a regression/migration backend only.
  #
  # It must not merely be *preferred* here, it must be the only backend
  # present. GemsimHIPDriver.is_active() gates itself on
  # TRITON_DEFAULT_BACKEND, but that does not make the in-tree amd driver
  # inactive, so both claim the same device. Triton's own selector is happy --
  # it consults TRITON_DEFAULT_BACKEND directly -- but any consumer that counts
  # active drivers is not: vLLM requires exactly one, and on finding two it
  # disables Triton outright, after which the run dies at the first use with
  #   AttributeError: module 'triton' has no attribute 'next_power_of_2'
  # having already retired 447 dispatches. Discovering only in-tree backends
  # leaves exactly one active driver for every consumer.
  export TRITON_BACKENDS_IN_TREE=1
  # Name the in-tree backend explicitly: the shared block above selects
  # gemsim_hip, which is no longer discovered here, and Triton rejects an
  # unknown TRITON_DEFAULT_BACKEND outright.
  export TRITON_DEFAULT_BACKEND=amd
  # Keep PYTHONPATH empty for upstream purity EXCEPT the read-only Triton
  # probe sitecustomize, which (a) names kernels through the launch log and
  # (b) optionally disables do_bench's L2-flush zero during autotune
  # (SAGR_TRITON_FAST_AUTOTUNE=1) -- the flush is the one-time-tuning wall
  # that consumed whole 12h lanes on 16.7M-workitem fill kernels.  Both
  # behaviors are gated by env and write no state into the engine.
  if [[ -n ${SAGR_TRITON_LAUNCH_LOG:-} || -n ${SAGR_TRITON_FAST_AUTOTUNE:-} ]]; then
    export PYTHONPATH="${ROOT}/tools/triton_launch_probe"
  else
    export PYTHONPATH=""
  fi
  # Empty allowlist: vllm/plugins/__init__.py loads a plugin only when its name
  # is in VLLM_PLUGINS, and an empty value parses to a list matching nothing.
  # This is the hard kill switch for both vllm.platform_plugins and
  # vllm.general_plugins, so no project operator can register even by accident.
  export VLLM_PLUGINS=
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_NO_USAGE_STATS=1
  export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((tp - 1)))
fi

# --- identity ---------------------------------------------------------------
{
  echo "# lane"
  echo "engine=${engine} tp=${tp} model=${model} dummy_weights=${dummy_weights}"
  echo "run_root=${SAGR_MANAGED_RUN_ROOT:-<unset>}"
  echo "topology=${HSA_MODEL_TOPOLOGY} gpu_nodes=$(( $(ls "${HSA_MODEL_TOPOLOGY}/nodes" | wc -l) - 1 ))"
  echo "# identity"
  echo "repo_head=$(git -C "$ROOT" rev-parse HEAD)"
  echo "rocm_systems_head=$(git -C "${ROOT}/projects/rocm-systems" rev-parse HEAD)"
  echo "product=${HEAD_PRODUCT}"
  echo "rocr_library_dir=${ROCR_LIBRARY_DIR}"
  echo "rocr_library_sha256=$(sha256sum "${ROCR_LIBRARY_DIR}/libhsa-runtime64.so.1.21.0" 2>/dev/null | cut -d' ' -f1)"
  echo "runtime_sha256=$(sha256sum "${RUNTIME_BUILD}/libself_amdgpu_runtime.so.0.8.0" | cut -d' ' -f1)"
  echo "model_lib_sha256=$(sha256sum "${HSA_MODEL_LIB}" | cut -d' ' -f1)"
  echo "gpu_archs=${GPU_ARCHS}"
  echo "sim_amdgpu_arch_sha256=$(sha256sum "${ROOT}/tools/sim_amdgpu_arch.sh" | cut -d' ' -f1)"
  echo "rocminfo_sha256=$(sha256sum "${TOOL_SHIM}/rocminfo" | cut -d' ' -f1)"
  echo "gem5=${SAGR_MANAGED_GEM5}"
  echo "gem5_sha256=$(sha256sum "${SAGR_MANAGED_GEM5}" | cut -d' ' -f1)"
  echo "gem5_base=${gem5_base:-${SAGR_MANAGED_GEM5}}"
  echo "gem5_base_sha256=$(sha256sum "${gem5_base:-${SAGR_MANAGED_GEM5}}" | cut -d' ' -f1)"
  echo "compute_units=${compute_units:-default}"
  echo "max_new_tokens=${max_new_tokens}"
  echo "hip_mode=${hip_mode}"
  echo "debug_layer_gate=${debug_layer_gate}"
  if (( debug_layer_gate )); then
    echo "layer_gate_output=${SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT}"
    echo "layer_gate_golden=${SAGR_QWEN35_SGLANG_LAYER_GATE_GOLDEN}"
    echo "operator_gate_golden=${SAGR_QWEN35_OPERATOR_GOLDEN}"
    echo "operator_gate_golden_metadata_sha256=$(sha256sum "${SAGR_QWEN35_OPERATOR_GOLDEN}/metadata.json" | cut -d' ' -f1)"
    echo "operator_gate_golden_results_sha256=$(sha256sum "${SAGR_QWEN35_OPERATOR_GOLDEN}/results.safetensors" | cut -d' ' -f1)"
    echo "layer_gate_sha256=$(sha256sum "${ROOT}/tools/qwen35_sglang_layer_gate/qwen35_sglang_layer_gate.py" | cut -d' ' -f1)"
    echo "triton_launch_log=${SAGR_TRITON_LAUNCH_LOG}"
  fi
  echo "hip_runtime_library=${HIP_RUNTIME_LIBRARY:-<loader>}"
  echo "fastcopy=HSA_ENABLE_DTIF_FAST_COPY=${HSA_ENABLE_DTIF_FAST_COPY} SAGR_HSAKMT_MODEL_FAST_COPY=${SAGR_HSAKMT_MODEL_FAST_COPY}"
  if [[ $engine == vllm ]]; then
    echo "# unchanged-upstream evidence"
    echo "vllm_head=$(git -C "${ROOT}/projects/vllm" rev-parse HEAD 2>/dev/null)"
    echo "vllm_dirty_files=$(git -C "${ROOT}/projects/vllm" status --porcelain -- . 2>/dev/null | wc -l)"
    echo "vllm_plugins_env=[${VLLM_PLUGINS}]"
    echo -n "gemsim_entry_points_in_site_packages="
    grep -l 'gemsim' "${PREFIX}"/lib/python3.12/site-packages/*.dist-info/entry_points.txt 2>/dev/null | wc -l
  else
    echo "sglang_dirty_files=$(git -C "${ROOT}/projects/sglang-0.5.17" status --porcelain -- . 2>/dev/null | wc -l)"
  fi
  echo "started=$(date -Is)"
  echo "--- run-identity-gate ---"
  identity_gate_args=(--format text)
  if (( product_hip )); then
    identity_gate_args+=(--require-active-product=libhsa-runtime64.so.1,libamdhip64.so.7)
  fi
  python3 tools/run_identity_gate.py "${identity_gate_args[@]}" 2>&1
} >"$log"
identity_gate_status=$?
if (( product_hip && identity_gate_status != 0 )); then
  printf 'product HIP identity gate failed; see %s\n' "$log" >&2
  exit 2
fi

# --- the run ----------------------------------------------------------------
# This host exposes a real NVIDIA RTX 5090 through WSL. With the
# simulator-aware AMD SMI provider on the library path, upstream platform
# selection sees ROCm *and* CUDA and refuses to activate two platforms. Masking
# NVML in a private mount namespace is isolation, not a framework change, and
# the goal forbids any NVIDIA fallback in an accepted run.
: > /tmp/empty-nvml.so

if [[ -n $capsule ]]; then
  # A capsule proves something about the lanes only if it runs in the lanes'
  # exact environment and isolation, which is why it is dispatched from here
  # rather than from a script of its own.
  #
  # SAGR_CAPSULE_ARGS: optional extra arguments appended to the capsule
  # invocation, split on whitespace (no quoting/escaping supported). Generic
  # mechanism so a capsule can expose modes (e.g. --probe NAME) without this
  # runner growing capsule-specific flags.
  read -r -a capsule_extra_args <<<"${SAGR_CAPSULE_ARGS:-}"
  unshare -r -m bash -c '
    mount --bind /tmp/empty-nvml.so /usr/lib/wsl/lib/libnvidia-ml.so.1
    exec python "$@"
  ' _ "$capsule" ${capsule_extra_args[@]+"${capsule_extra_args[@]}"} >>"$log" 2>&1
  status=$?
elif [[ $engine == sglang ]]; then
  sglang_args=(
    --tp-size "$tp" --attention-backend "${SAGR_ATTENTION_BACKEND:-aiter}"
    --context-length 16 --max-total-tokens 16 --max-mamba-cache-size 5
    --max-new-tokens "$max_new_tokens" --seed 0 --watchdog-timeout 86400
    --dist-timeout 86400 --model-path "$model"
  )
  (( dummy_weights )) && sglang_args+=(--load-format dummy)
  unshare -r -m bash -c '
    mount --bind /tmp/empty-nvml.so /usr/lib/wsl/lib/libnvidia-ml.so.1
    exec python examples/sglang/qwen35_inference.py "$@"
  ' _ "${sglang_args[@]}" >>"$log" 2>&1
  status=$?
else
  vllm_args=(
    --tp-size "$tp" --model-path "$model"
    --context-length 16 --max-new-tokens "$max_new_tokens" --max-num-seqs 1
    --seed 0
    # vLLM sizes every cache pool from the utilization fraction of the
    # device's reported memory; on the simulated 287 GB device the 0.30
    # default dedicates ~84 GB to caches, and zeroing that mamba pool is
    # what dominated zcode-vllm-tp1-v1 past its 12 h ceiling (643 kernels
    # of 16.7M workitems before the dummy profile run even finished).  A
    # 16-token context with one sequence needs almost nothing.
    --gpu-memory-utilization "${SAGR_VLLM_GPU_MEM_UTIL:-0.30}"
  )
  # The driver's RPC into workers (sample_tokens and friends) times out
  # after VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (default 300s).  Under the
  # simulator the sampler plus its TP2 collectives can exceed that wall
  # clock while being perfectly alive -- vLLM TP2 once reached the very
  # last sampling step of generation and died only on this timeout.
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${SAGR_VLLM_RPC_TIMEOUT_SECONDS:-300}"
  # The NCCL watchdog separately kills any single collective that outlives
  # ParallelConfig.distributed_timeout_seconds (default 600s); the example
  # exposes it as --distributed-timeout.  Under the simulator the TP2
  # logits all-gather is a legitimately multi-hour collective.
  vllm_args+=(--distributed-timeout "${SAGR_VLLM_DIST_TIMEOUT_SECONDS:-600}")
  (( dummy_weights )) && vllm_args+=(--load-format dummy)
  # A real file, not a heredoc: with tensor_parallel_size > 1 vLLM spawns
  # workers through multiprocessing, and spawn re-imports the parent __main__
  # from its recorded path. A stdin-fed program records "<stdin>", so every
  # worker dies with FileNotFoundError before touching the GPU.
  unshare -r -m bash -c '
    mount --bind /tmp/empty-nvml.so /usr/lib/wsl/lib/libnvidia-ml.so.1
    exec python examples/vllm/qwen35_inference.py "$@"
  ' _ "${vllm_args[@]}" >>"$log" 2>&1
  status=$?
fi

echo "finished=$(date -Is) status=${status}" >>"$log"
exit "$status"
