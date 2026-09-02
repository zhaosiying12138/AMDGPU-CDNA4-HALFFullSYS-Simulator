#!/usr/bin/env bash
# Turn a phase-A maps capture (/tmp/vm_phaseA/maps.json: JSON list of paths
# mapped by a live TP2 lane process tree) into the publish syslibs list.
#
# Keeps every mapped file under the lane root that is not already delivered
# by the base/python/model streams, plus anything under /opt or /usr/local.
# System paths (/usr/lib, /lib) are dropped: the Ubuntu 24.04 guest provides
# its own, and host binaries only need symbol versions <= guest glibc 2.39.
#
# Usage: gen_phaseA_libs.sh [maps.json] > phaseA_libs.list
set -euo pipefail

MAPS="${1:-/tmp/vm_phaseA/maps.json}"
ROOT=/home/zhaosiying/zcode-lane

python3 - "$MAPS" <<'EOF' |
import json, sys
for p in json.load(open(sys.argv[1])):
    print(p)
EOF
grep -E "^($ROOT|/home/zhaosiying/amdgpu-sim|/home/zhaosiying/.cache|/opt|/usr/local)/" |
grep -v -E "^($ROOT|/home/zhaosiying/amdgpu-sim)/(env/conda|projects/sglang|env/sglang-overlay|models|artifacts/zcode-cache)" |
grep -v -E "^($ROOT|/home/zhaosiying/amdgpu-sim)/(build/rocr-stage-zcode|projects/self-amdgpu-runtime/build|env/rocm|projects/gem5|artifacts/topology)" |
sort -u
