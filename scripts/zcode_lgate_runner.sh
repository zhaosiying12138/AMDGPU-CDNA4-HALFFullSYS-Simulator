#!/usr/bin/env bash
# Parameterized layer-gate lane runner.
#   LANE_NAME      lane directory under artifacts/lanes
#   RECORD_ALL=1   gate survey mode (record every mismatch, continue)
#   ATTN           attention backend (default aiter; triton is the upstream
#                  AMD recommendation per GOAL.md's SGLang lane text)
# SAGR_MANAGED_RUN_ROOT must be forwarded through env -i or the lane's
# simulators escape to the global default run root and lane reaping plus
# ownership confinement stop working.
LANE_NAME="${LANE_NAME:-zcode-lgate}"
ATTN="${ATTN:-aiter}"
RECORD_ALL_ENV=()
if [[ -n "${RECORD_ALL:-}" ]]; then
  RECORD_ALL_ENV=(SAGR_QWEN35_LAYER_GATE_RECORD_ALL=1)
fi
exec env -i HOME=/home/zhaosiying TERM=dumb \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  SAGR_MANAGED_RUN_ROOT="/tmp/sagr-lane-${LANE_NAME}" \
  SAGR_ROCR_LIBRARY_DIR=/home/zhaosiying/zcode-lane/build/rocr-stage-zcode/lib \
  SAGR_ATTENTION_BACKEND="${ATTN}" \
  "${RECORD_ALL_ENV[@]}" \
  bash /home/zhaosiying/zcode-lane/scripts/run_engine_lane.sh \
  --engine sglang --tp 1 --debug-layer-gate --fast \
  "/home/zhaosiying/zcode-lane/artifacts/lanes/${LANE_NAME}/lane.log"
