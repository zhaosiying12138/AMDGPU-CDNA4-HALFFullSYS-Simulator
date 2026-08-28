#!/bin/bash
# Attach gdb to the scheduler child (a NEW python process appearing after the
# parent) and dump the native stack when it takes SIGSEGV/SIGABRT.
PARENT="$1"
for i in $(seq 1 90); do
  for P in $(pgrep -f "python.*mini_engine_construct\|qwen35_inference"); do
    if [ "$P" != "$PARENT" ] && [ -d /proc/$P ]; then
      echo "attaching child $P"
      timeout 400 gdb -p "$P" -batch \
        -ex "set pagination off" \
        -ex "handle SIGSEGV stop print nopass" \
        -ex "handle SIGABRT stop print nopass" \
        -ex "continue" \
        -ex \"echo ===STACK===\\n\" \
        -ex "bt 25" \
        -ex "info threads" \
        > /home/zhaosiying/zcode-lane/artifacts/hybrid-cta-capsule-v2/mini-engine/gdb_child.log 2>&1
      exit 0
    fi
  done
  sleep 1
done
echo "no child found"
