#!/usr/bin/env bash
# Decode-gate lane: prefill + both decode steps gated against the decode4
# golden row-by-row (record-all survey over all 24 layers x 2 decode rows).
LANE_NAME="${LANE_NAME:-zcode-decode}"
exec env -i HOME=/home/zhaosiying TERM=dumb \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  SAGR_MANAGED_RUN_ROOT="/tmp/sagr-lane-${LANE_NAME}" \
  SAGR_ROCR_LIBRARY_DIR=/home/zhaosiying/zcode-lane/build/rocr-stage-zcode/lib \
  SAGR_QWEN35_SGLANG_LAYER_GATE_RECORD_ALL=1 \
  SAGR_QWEN35_SGLANG_LAYER_GATE_GOLDEN=/home/zhaosiying/amdgpu-sim/artifacts/qwen35-nvidia-golden/20260812-prefill2-max24-v1 \
  SAGR_QWEN35_DECODE_GATE_GOLDEN=/home/zhaosiying/zcode-lane/artifacts/qwen35-nvidia-golden/20260812-decode4-max24-v1 \
  SAGR_ATTENTION_BACKEND=triton \
  bash /home/zhaosiying/zcode-lane/scripts/run_engine_lane.sh \
  --engine sglang --tp 1 --debug-layer-gate --fast --max-new-tokens 2 \
  "/home/zhaosiying/zcode-lane/artifacts/lanes/${LANE_NAME}/lane.log"
