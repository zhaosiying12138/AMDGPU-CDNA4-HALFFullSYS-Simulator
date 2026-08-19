#!/usr/bin/env bash
# Parameterized layer-gate lane runner: LANE_NAME selects the lane directory.
# SAGR_MANAGED_RUN_ROOT must be forwarded through env -i or the lane's
# simulators escape to the global default run root and lane reaping plus
# ownership confinement stop working.
LANE_NAME="${LANE_NAME:-zcode-lgate}"
exec env -i HOME=/home/zhaosiying TERM=dumb \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  SAGR_MANAGED_RUN_ROOT="/tmp/sagr-lane-${LANE_NAME}" \
  SAGR_ROCR_LIBRARY_DIR=/home/zhaosiying/zcode-lane/build/rocr-stage-0401e8cd/lib \
  bash /home/zhaosiying/zcode-lane/scripts/run_engine_lane.sh \
  --engine sglang --tp 1 --debug-layer-gate --fast \
  "/home/zhaosiying/zcode-lane/artifacts/lanes/${LANE_NAME}/lane.log"
