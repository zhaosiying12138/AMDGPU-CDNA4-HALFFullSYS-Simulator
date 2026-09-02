#!/bin/bash
# Self-contained hybrid-crash reproducer with native stack capture:
# 1. launch the TP1 layer-gate lane under hybridfastwrap (60-90s to crash)
# 2. stalk the spawned scheduler child
# 3. gdb-attach to the child, stop on SIGSEGV/SIGABRT, dump native bt
# All artifacts under OUT (workspace, /tmp-cleaner-proof).
set -u
cd /home/zhaosiying/zcode-lane
OUT=artifacts/hybrid-cta-capsule-v2/mini-engine
mkdir -p "$OUT"
for p in $(pgrep gem5 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
pkill -9 -f qwen35_inference 2>/dev/null
sleep 1
unset SAGR_LANE_PYTHONPATH_PREFIX PYTHONFAULTHANDLER
export SAGR_MANAGED_GEM5=/home/zhaosiying/zcode-gem5-hybrid2/build/VEGA_X86/gem5.opt.hybridfastwrap2
export SAGR_MANAGED_GEM5_CONFIG=/home/zhaosiying/zcode-gem5-hybrid2/configs/example/gemsim/host_dispatch.py
export SAGR_MANAGED_RUN_ROOT=/tmp/sagr-lg-stalk
rm -rf /tmp/sagr-lg-stalk
export SAGR_DBG_OUT=yes
nohup bash scripts/run_engine_lane.sh --engine sglang --tp 1 --fast \
    --debug-layer-gate "$OUT/stalk-lane.log" > "$OUT/stalk-runner.out" 2>&1 &
LANE=$!
echo "lane pid $LANE" >> "$OUT/stalk.log"
sleep 5
PARENT=$(pgrep -f "qwen35_inference" | head -1)
echo "parent $PARENT" >> "$OUT/stalk.log"
CHILD=""
for i in $(seq 1 150); do
  # mp-spawn children run "python -c from multiprocessing...spawn_main..."
  for P in $(pgrep -f "spawn_main"); do
    if [ "$P" != "$PARENT" ]; then CHILD=$P; break 2; fi
  done
  sleep 1
done
echo "child $CHILD" >> "$OUT/stalk.log"
if [ -n "$CHILD" ]; then
  if true; then timeout 420 gdb -p "$CHILD" -batch \
      -ex "set pagination off" \
      -ex "handle SIGSEGV stop print nopass" \
      -ex "handle SIGABRT stop print nopass" \
      -ex "continue" \
      -ex "bt 30" \
      > "$OUT/gdb_native.log" 2>&1
  echo "gdb rc=$?" >> "$OUT/stalk.log"; fi
else
  echo "no child found" >> "$OUT/stalk.log"
fi
wait $LANE 2>/dev/null
echo "done" >> "$OUT/stalk.log"
