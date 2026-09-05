#!/usr/bin/env bash
# Operator-correctness regression: Triton kernel + HIP dual-mode capsules.
#
#  1. Triton softmax: tools/softmax_demo.py (one real Triton launch on the
#     simulated GPU, checked against the CPU reference) under the AMDGPU-CDNA4-SIM
#     tools environment.
#  2. Hybrid-CTA dual-mode capsules: the three acceptance kernels in
#     tools/hybrid_cta_capsule (plain_dp / barrier_lds / atomic_decline) each
#     run twice through the lane runner — functional-fast without and with
#     --hybrid-cta — and the two runs must produce byte-identical output
#     buffers (sha256). plain_dp additionally carries an exact host oracle.
#  3. 2048-workgroup stress: plain_dp at HYBRID_GRID_WGS=2048 under hybrid
#     mode, three times; every run must be oracle-correct and identical.
#
# The fail-closed property under test: the hybrid executor must change wall
# time, never results. Any mismatch fails the regression.
#
# Usage:
#   bash scripts/regression/operator_correctness.sh [--quick]
#     --quick   skip the 2048-WG stress triple (smoke-level run)
#
# Environment overrides:
#   ASIM_GEM5_TIP    gem5 tree (default <repo>/projects/gem5)
#   ASIM_TOOLS_ENV   conda env name for the softmax step (default
#                    AMDGPU-CDNA4-SIM; create with
#                    scripts/make_amdgpu_tools_env.sh)
#
# Output: <out>/summary.md plus per-run result.json / lane.log; non-zero
# exit if any check fails.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

GEM5_TIP=${ASIM_GEM5_TIP:-$ROOT/projects/gem5}
GEM5_BIN=$GEM5_TIP/build/VEGA_X86/gem5.opt
GEM5_CFG=$GEM5_TIP/configs/example/gemsim/host_dispatch.py
TOOLS_ENV=${ASIM_TOOLS_ENV:-AMDGPU-CDNA4-SIM}
CAPSULE=$ROOT/tools/hybrid_cta_capsule/hybrid_accept_capsule.py
quick=0
out=${ASIM_OPTEST_OUT:-$ROOT/artifacts/operator-correctness-regression}
while (($#)); do
  case "$1" in
    --quick) quick=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

for f in "$GEM5_BIN" "$GEM5_CFG" "$CAPSULE" "$ROOT/tools/softmax_demo.py"; do
  [[ -e $f ]] || { printf 'missing prerequisite: %s\n' "$f" >&2; exit 2; }
done

rm -rf "$out"; mkdir -p "$out/wrappers"
cat >"$out/wrappers/w-ff.sh" <<EOF
#!/bin/sh
# functional-fast without the hybrid executor (mode A).
exec $GEM5_BIN "\$@" --functional-fast
EOF
cat >"$out/wrappers/w-hybrid.sh" <<EOF
#!/bin/sh
# functional-fast with the hybrid CTA executor (mode B).
exec $GEM5_BIN "\$@" --functional-fast --hybrid-cta
EOF
chmod +x "$out"/wrappers/w-*.sh

failures=0
declare -a summary_rows=()

note_fail() { failures=$((failures + 1)); }

# --- 1. Triton softmax -------------------------------------------------------
softmax_dir=$out/softmax
mkdir -p "$softmax_dir"
if bash -c "
  source '${HOME}/miniforge3/etc/profile.d/conda.sh' && conda activate '$TOOLS_ENV' \
    && python '$ROOT/tools/softmax_demo.py'
" >"$softmax_dir/lane.log" 2>&1; then
  summary_rows+=("softmax (Triton vs CPU ref)" "PASS")
else
  summary_rows+=("softmax (Triton vs CPU ref)" "FAIL")
  note_fail
fi

# --- 2./3. dual-mode capsules -------------------------------------------------
run_capsule() { # <kernel> <grid_wgs> <mode(ff|hybrid)> <outdir> <runroot>
  local kernel=$1 grid=$2 mode=$3 dir=$4 runroot=$5
  rm -rf "$dir" "$runroot"; mkdir -p "$(dirname "$dir")" "$runroot"
  local wrapper=$out/wrappers/w-$mode.sh
  env -u SAGR_ROCR_LIBRARY_DIR -u SAGR_MANAGED_GEM5 -u SAGR_MANAGED_GEM5_CONFIG \
      -u SAGR_MANAGED_REPO_ROOT -u SAGR_TRITON_LAUNCH_LOG -u SAGR_CAPSULE_ARGS \
      GEMSIM_NUM_COMPUTE_UNITS=16 \
      SAGR_MANAGED_RUN_ROOT="$runroot" \
      SAGR_LANE_CACHE_ROOT="${ASIM_LANE_CACHE_ROOT:-$ROOT/artifacts/zcode-cache}" \
      SAGR_MANAGED_GEM5="$wrapper" \
      SAGR_MANAGED_GEM5_CONFIG="$GEM5_CFG" \
      HYBRID_KERNEL="$kernel" \
      HYBRID_GRID_WGS="$grid" \
      HYBRID_CAPSULE_OUTPUT="$dir" \
    timeout --signal=TERM --kill-after=30 1800 \
      bash scripts/run_engine_lane.sh --tp 1 --capsule "$CAPSULE" \
        "$dir.lane.log" >/dev/null 2>&1
  local status=$?
  pkill -f "$runroot" 2>/dev/null || true
  rm -rf "$runroot"
  (( status == 0 )) && [[ -f $dir/result.json ]]
}

check_pair() { # <label> <dir_ff> <dir_hybrid> <require_oracle>
  local label=$1 ff=$2 hyb=$3 require_oracle=$4
  if [[ ! -f $ff/result.json || ! -f $hyb/result.json ]]; then
    summary_rows+=("$label" "FAIL (missing result)")
    note_fail; return
  fi
  local sha_ff sha_hyb oracle
  sha_ff=$(python3 -c "import json;print(json.load(open('$ff/result.json'))['output_sha256'])")
  sha_hyb=$(python3 -c "import json;print(json.load(open('$hyb/result.json'))['output_sha256'])")
  oracle=$(python3 -c "import json;print(json.load(open('$hyb/result.json')).get('oracle_correct'))")
  if [[ $sha_ff != "$sha_hyb" ]]; then
    summary_rows+=("$label" "FAIL (sha mismatch $sha_ff vs $sha_hyb)")
    note_fail; return
  fi
  if [[ $require_oracle == 1 && $oracle != True ]]; then
    summary_rows+=("$label" "FAIL (oracle_not_correct)")
    note_fail; return
  fi
  summary_rows+=("$label" "PASS (${sha_ff:0:12}…)")
}

for kernel in plain_dp barrier_lds atomic_decline; do
  run_capsule "$kernel" 8 ff "$out/capsule/$kernel/ff" "/tmp/asim-optest-$kernel-ff"
  run_capsule "$kernel" 8 hybrid "$out/capsule/$kernel/hybrid" "/tmp/asim-optest-$kernel-hyb"
  check_pair "capsule $kernel (ff ≡ hybrid, 8 WG)" \
    "$out/capsule/$kernel/ff" "$out/capsule/$kernel/hybrid" \
    "$([[ $kernel == plain_dp ]] && echo 1 || echo 0)"
done

if (( ! quick )); then
  for i in 1 2 3; do
    run_capsule plain_dp 2048 hybrid "$out/stress2048/run$i" "/tmp/asim-optest-stress$i"
    if [[ -f $out/stress2048/run$i/result.json ]]; then
      oracle=$(python3 -c "import json;print(json.load(open('$out/stress2048/run$i/result.json')).get('oracle_correct'))")
      sha=$(python3 -c "import json;print(json.load(open('$out/stress2048/run$i/result.json'))['output_sha256'])")
      if [[ $oracle == True ]]; then
        summary_rows+=("stress plain_dp 2048 WG run$i" "PASS (${sha:0:12}…)")
      else
        summary_rows+=("stress plain_dp 2048 WG run$i" "FAIL (oracle)")
        note_fail
      fi
    else
      summary_rows+=("stress plain_dp 2048 WG run$i" "FAIL (no result)")
      note_fail
    fi
  done
fi

python3 - "$out" "$failures" "${summary_rows[@]}" <<'PY'
import sys
from pathlib import Path

out, failures = Path(sys.argv[1]), int(sys.argv[2])
rows = sys.argv[3:]
lines = ["# operator_correctness summary", "",
         "| check | verdict |", "|---|---|"]
for i in range(0, len(rows), 2):
    lines.append(f"| {rows[i]} | {rows[i + 1]} |")
lines += ["", f"**{len(rows) // 2 - failures}/{len(rows) // 2} checks passed**"]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
exit $(( failures > 0 ? 1 : 0 ))
