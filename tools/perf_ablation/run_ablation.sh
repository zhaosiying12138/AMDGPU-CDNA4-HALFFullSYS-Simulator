#!/usr/bin/env bash
# Perf-ablation harness: run one SGLang TP1 0.8B 1-token lane under a named
# configuration and capture everything the parser needs.
#
# Configurations (see docs/blog/gem5-performance-chapter.md):
#   c2-status-quo    warm cache  + correctness autotune + fastcopy fast   (= accepted lanes)
#   c1-legacy-copy   warm cache  + correctness autotune + fastcopy legacy
#   c0p-cold-correct cold cache  + correctness autotune + fastcopy legacy
#   c0-cold-device   cold cache  + device autotune       + fastcopy legacy (slowest baseline)
#
# Usage: run_ablation.sh <config> [timeout-seconds]
# Output: artifacts/perf-ablation-2026-08/<config>/{lane.log,runroot/,timeout.marker}
set -uo pipefail

ROOT=/home/zhaosiying/zcode-lane
OUT="${ROOT}/artifacts/perf-ablation-2026-08"
CONFIG="${1:?usage: run_ablation.sh <config> [timeout-seconds]}"
TIMEOUT_S="${2:-10800}"

case "$CONFIG" in
  c2-status-quo)
    export SAGR_LANE_CACHE_ROOT="${ROOT}/artifacts/zcode-cache"
    export SAGR_LANE_FASTCOPY_MODE=fast
    export SAGR_LANE_AUTOTUNE_MODE=correctness ;;
  c1-legacy-copy)
    export SAGR_LANE_CACHE_ROOT="${ROOT}/artifacts/zcode-cache"
    export SAGR_LANE_FASTCOPY_MODE=legacy
    export SAGR_LANE_AUTOTUNE_MODE=correctness ;;
  c0p-cold-correct)
    export SAGR_LANE_CACHE_ROOT="${OUT}/cold-cache"
    rm -rf "${OUT}/cold-cache"
    export SAGR_LANE_FASTCOPY_MODE=legacy
    export SAGR_LANE_AUTOTUNE_MODE=correctness ;;
  c0-cold-device)
    export SAGR_LANE_CACHE_ROOT="${OUT}/cold-cache"
    rm -rf "${OUT}/cold-cache"
    export SAGR_LANE_FASTCOPY_MODE=legacy
    export SAGR_LANE_AUTOTUNE_MODE=device ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2 ;;
esac

DIR="${OUT}/${CONFIG}"
rm -rf "$DIR"
mkdir -p "$DIR/runroot"

# The lane honors several SAGR_* inheritance points (ROCr stage, gem5
# selection); a leaked value from an interactive shell silently swaps the
# preloaded ROCr stage and the run dies at hsa_init 4104 (seen once: the
# identity header recorded rocr-stage-0401e8cd).  Scrub everything the lane
# is meant to default itself.
unset SAGR_ROCR_LIBRARY_DIR SAGR_MANAGED_GEM5 SAGR_MANAGED_GEM5_CONFIG \
      SAGR_MANAGED_REPO_ROOT SAGR_TRITON_LAUNCH_LOG \
      SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT SAGR_QWEN35_OPERATOR_GOLDEN \
      SAGR_QWEN35_SGLANG_LAYER_GATE_GOLDEN SAGR_CAPSULE_ARGS
# AF_UNIX pathname limit is 108 bytes: the artifacts path is far too long
# for the managed session's bridge socket (gem5 aborts with "endpoint is too
# long"), so keep the live runroot short and archive it after the run.
RUNROOT="/tmp/sagr-abl-${CONFIG}"
rm -rf "$RUNROOT"
export SAGR_MANAGED_RUN_ROOT="$RUNROOT"

echo "[ablation] config=$CONFIG timeout=${TIMEOUT_S}s runroot=$DIR"
timeout --signal=TERM --kill-after=60 "$TIMEOUT_S" \
  bash "${ROOT}/scripts/run_engine_lane.sh" \
    --engine sglang --tp 1 --fast --max-new-tokens 1 \
    "${DIR}/lane.log"
status=$?
if (( status == 124 )); then
  echo "[ablation] TIMEOUT after ${TIMEOUT}s" | tee "${DIR}/timeout.marker"
fi
echo "[ablation] done status=$status"
exit "$status"
