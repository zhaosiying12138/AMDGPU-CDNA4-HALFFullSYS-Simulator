#!/bin/bash
cd /home/zhaosiying/zcode-lane
export SAGR_MANAGED_GEM5_CONFIG=/home/zhaosiying/zcode-gem5-hybrid2/configs/example/gemsim/host_dispatch.py
export SAGR_MANAGED_RUN_ROOT=/home/zhaosiying/rdbg
rm -rf /home/zhaosiying/rdbg
export SAGR_MANAGED_GEM5=/home/zhaosiying/zcode-gem5-hybrid2/build/VEGA_X86/gem5.opt.hybridfastwrap2
exec bash scripts/run_engine_lane.sh --engine sglang --tp 1 --capsule tools/hybrid_cta_capsule/mini_engine_construct.py artifacts/hybrid-cta-capsule-v2/mini-engine/dbg-lane.log
