#!/usr/bin/env bash
# Token-gate lane runner: plain inference (no layer gate), Triton attention
# backend (the upstream AMD recommendation per GOAL.md), 2 new tokens
# compared by the inference script against the frozen token golden.
LANE_NAME="${LANE_NAME:-zcode-token}"
exec env -i HOME=/home/zhaosiying TERM=dumb \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  SAGR_MANAGED_RUN_ROOT="/tmp/sagr-lane-${LANE_NAME}" \
  SAGR_ROCR_LIBRARY_DIR=/home/zhaosiying/zcode-lane/build/rocr-stage-zcode/lib \
  SAGR_ATTENTION_BACKEND=triton \
  SAGR_SGLANG_USE_AITER=0 \
  bash /home/zhaosiying/zcode-lane/scripts/run_engine_lane.sh \
  --engine sglang --tp 1 --fast --max-new-tokens 2 \
  "/home/zhaosiying/zcode-lane/artifacts/lanes/${LANE_NAME}/lane.log"
