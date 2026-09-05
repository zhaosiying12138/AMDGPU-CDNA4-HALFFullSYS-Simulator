#!/usr/bin/env bash
# Pure-performance regression bench: SGLang TP1 · Qwen3.5-0.8B · 1 token.
#
# Measures end-to-end host wall time plus the engine's own phase timings
# (load_weight / kv_cache_allocation / scheduler_e2e / request e2e) for a
# quick performance-regression comparison between simulator configurations.
# It is deliberately perf-only: the lane's built-in token gate verdict is
# recorded as informational data, not asserted — correctness coverage lives
# in scripts/regression/operator_correctness.sh.
#
# Arms (run sequentially; the host should otherwise be idle):
#   legacy    current gem5 + --functional-fast, legacy DTIF copy
#             (SAGR_LANE_FASTCOPY_MODE=legacy), adaptive idle park disabled,
#             per-instruction progress tracking re-enabled — i.e. every
#             env-switchable optimization except the timing collapse is OFF.
#   full      current gem5 + --functional-fast --hybrid-cta, fast DTIF copy,
#             lane-default idle park — the accepted fast configuration.
#   baseline  (only with --with-baseline) bugfix-only gem5 binary
#             ($ASIM_GEM5_BASE, no KMT mapping cache) + --functional-fast +
#             legacy copy: the "all bugfixes, env optimizations off" anchor.
#             The fully-accurate (no --functional-fast) baseline is NOT part
#             of this quick bench: it is known to hit the KMT scratch
#             admission race on slow timing configurations (see the perf
#             blog, limitations section) and can hang for hours.
#
# Configuration matches the 2026-09 ablation campaign: CU16 (the accepted
# 0.8B text-gate geometry) and the shared warm Triton cache.
#
# Usage:
#   bash scripts/regression/perf_bench.sh [--with-baseline] [--tokens N]
#        [--timeout-legacy S] [--timeout-full S] [--out DIR]
#
# Environment overrides:
#   ASIM_GEM5_TIP   tree containing build/VEGA_X86/gem5.opt (default:
#                   <repo>/projects/gem5)
#   ASIM_GEM5_BASE  bugfix-only gem5 tree for the optional baseline arm
#                   (default: /home/zhaosiying/zcode-gem5-base at 8cd1db918)
#
# Output: <out>/<arm>/{lane.log,metrics.json,meta.txt,exit.status} and
#         <out>/summary.md   (a markdown table over all arms)
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

GEM5_TIP=${ASIM_GEM5_TIP:-$ROOT/projects/gem5}
GEM5_BASE=${ASIM_GEM5_BASE:-/home/zhaosiying/zcode-gem5-base}
TIP_BIN=$GEM5_TIP/build/VEGA_X86/gem5.opt
TIP_CFG=$GEM5_TIP/configs/example/gemsim/host_dispatch.py
BASE_BIN=$GEM5_BASE/build/VEGA_X86/gem5.opt
BASE_CFG=$GEM5_BASE/configs/example/gemsim/host_dispatch.py

with_baseline=0
tokens=1
timeout_legacy=9000
timeout_full=3600
out=${ASIM_PERF_BENCH_OUT:-$ROOT/artifacts/perf-bench-regression}
while (($#)); do
  case "$1" in
    --with-baseline) with_baseline=1; shift ;;
    --tokens) tokens=${2:?}; shift 2 ;;
    --timeout-legacy) timeout_legacy=${2:?}; shift 2 ;;
    --timeout-full) timeout_full=${2:?}; shift 2 ;;
    --out) out=${2:?}; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

for f in "$TIP_BIN" "$TIP_CFG"; do
  [[ -e $f ]] || { printf 'missing TIP prerequisite: %s\n' "$f" >&2; exit 2; }
done
if (( with_baseline )); then
  for f in "$BASE_BIN" "$BASE_CFG"; do
    [[ -e $f ]] || { printf 'missing BASE prerequisite: %s (set ASIM_GEM5_BASE or drop --with-baseline)\n' "$f" >&2; exit 2; }
  done
fi
if [[ ! -d $ROOT/artifacts/zcode-cache/triton ]]; then
  printf 'WARNING: warm Triton cache %s is absent; the first arm will pay cold-cache compilation.\n' \
    "$ROOT/artifacts/zcode-cache/triton" >&2
fi

rm -rf "$out"; mkdir -p "$out/wrappers"

# --- wrappers: managed_session scrubs env, so the knobs ride the wrapper ----
cat >"$out/wrappers/w-legacy.sh" <<EOF
#!/bin/sh
# perf_bench legacy arm: every env-switchable optimization off.
export GEMSIM_IDLE_PARK_S=0
export GEMSIM_PROGRESS_INTERVAL=20000000
exec $TIP_BIN "\$@" --functional-fast
EOF
cat >"$out/wrappers/w-full.sh" <<EOF
#!/bin/sh
# perf_bench full arm: the accepted fast configuration.
exec $TIP_BIN "\$@" --functional-fast --hybrid-cta
EOF
cat >"$out/wrappers/w-baseline.sh" <<EOF
#!/bin/sh
# perf_bench baseline arm: bugfix-only binary, env optimizations off.
export GEMSIM_IDLE_PARK_S=0
export GEMSIM_PROGRESS_INTERVAL=20000000
exec $BASE_BIN "\$@" --functional-fast
EOF
chmod +x "$out"/wrappers/w-*.sh

kill_runroot_strays() { # <runroot>: a timed-out lane must not pollute the next arm
  local runroot=$1 pids pid
  for round in TERM KILL; do
    pids=$(pgrep -f "$runroot" 2>/dev/null || true)
    [[ -z $pids ]] && return 0
    for pid in $pids; do kill -"$round" "$pid" 2>/dev/null || true; done
    sleep 3
  done
}

count_dispatches() { # <runroot>
  local total=0 n
  for f in "$1"/self-amdgpu-opencl-run.*/dispatch-trace.jsonl; do
    [[ -f $f ]] || continue
    n=$(wc -l <"$f")
    total=$((total + n))
  done
  printf '%s' "$total"
}

run_arm() { # <name> <wrapper> <config> <fastcopy> <timeout_s>
  local name=$1 wrapper=$2 config=$3 fastcopy=$4 timeout_s=$5
  local dir=$out/$name
  mkdir -p "$dir"
  local runroot=/tmp/asim-perfbench-$name
  rm -rf "$runroot"; mkdir -p "$runroot"

  printf '[perf-bench] arm=%s fastcopy=%s tokens=%s timeout=%ss %s\n' \
    "$name" "$fastcopy" "$tokens" "$timeout_s" "$(date -Is)"

  {
    echo "wrapper=$wrapper"
    echo "config=$config"
    echo "gem5_binary_sha256=$(sha256sum "$(grep -o 'exec [^ ]*/gem5.opt' "$wrapper" | head -1 | cut -d' ' -f2)" | cut -d' ' -f1)"
    echo "gem5_git_rev=$(git -C "$(dirname "$(dirname "$(grep -o 'exec [^ ]*/gem5.opt' "$wrapper" | head -1 | cut -d' ' -f2)")")" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "repo_head=$(git -C "$ROOT" rev-parse HEAD)"
    echo "fastcopy_env=$fastcopy"
    echo "compute_units=16"
    date -Is
  } >"$dir/meta.txt"

  local t0 t1 status
  t0=$(date +%s)
  env -u SAGR_ROCR_LIBRARY_DIR -u SAGR_MANAGED_GEM5 -u SAGR_MANAGED_GEM5_CONFIG \
      -u SAGR_MANAGED_REPO_ROOT -u SAGR_TRITON_LAUNCH_LOG \
      -u SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT -u SAGR_QWEN35_OPERATOR_GOLDEN \
      -u SAGR_QWEN35_SGLANG_LAYER_GATE_GOLDEN -u SAGR_CAPSULE_ARGS \
      GEMSIM_NUM_COMPUTE_UNITS=16 \
      SAGR_MANAGED_RUN_ROOT="$runroot" \
      SAGR_LANE_CACHE_ROOT="${ASIM_LANE_CACHE_ROOT:-$ROOT/artifacts/zcode-cache}" \
      SAGR_LANE_FASTCOPY_MODE="$fastcopy" \
      SAGR_MANAGED_GEM5="$wrapper" \
      SAGR_MANAGED_GEM5_CONFIG="$config" \
    timeout --signal=TERM --kill-after=60 "$timeout_s" \
      bash scripts/run_engine_lane.sh \
        --engine sglang --tp 1 --max-new-tokens "$tokens" \
        "$dir/lane.log"
  status=$?
  t1=$(date +%s)
  echo "$status" >"$dir/exit.status"
  (( status == 124 )) && printf 'TIMEOUT after %ss\n' "$timeout_s" >"$dir/timeout.marker"
  printf '[perf-bench] arm=%s lane_status=%s wall=%ss\n' "$name" "$status" "$((t1 - t0))"

  local dispatches
  dispatches=$(count_dispatches "$runroot")
  kill_runroot_strays "$runroot"
  rm -rf "$runroot"

  python3 - "$dir" "$name" "$((t1 - t0))" "$status" "$dispatches" <<'PY'
import json, re, sys
from pathlib import Path

arm_dir, name, wall, status, dispatches = (
    Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
log = (arm_dir / "lane.log").read_text(errors="replace") if (arm_dir / "lane.log").exists() else ""

def grab(pattern, cast=float):
    m = re.search(pattern, log)
    return cast(m.group(1)) if m else None

metrics = {
    "schema": "amdgpu-sim.perf-bench.v1",
    "arm": name,
    "lane_status": int(status),
    "wall_s": wall,
    "load_weight_s": grab(r"load_weight=([\d.]+)"),
    "kv_cache_allocation_s": grab(r"kv_cache_allocation=([\d.]+)"),
    "scheduler_e2e_s": grab(r"scheduler_e2e=([\d.]+)"),
    "request_e2e_s": grab(r'"e2e_latency": ([\d.]+)'),
    "retired_dispatches": dispatches,
    "timed_out": (arm_dir / "timeout.marker").exists(),
    # Informational only: this bench does not gate on correctness.
    "token_gate_correct": bool(re.search(r'token_golden=\{.*"correct":\s*true', log)),
}
(arm_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
PY
  return 0
}

run_arm legacy  "$out/wrappers/w-legacy.sh"  "$TIP_CFG"  legacy "$timeout_legacy"
run_arm full    "$out/wrappers/w-full.sh"    "$TIP_CFG"  fast   "$timeout_full"
(( with_baseline )) && run_arm baseline "$out/wrappers/w-baseline.sh" "$BASE_CFG" legacy 9000

python3 - "$out" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for arm in ("baseline", "legacy", "full"):
    f = out / arm / "metrics.json"
    if f.exists():
        rows.append((arm, json.loads(f.read_text())))

def fmt(m, k, scale=1.0):
    v = m.get(k)
    return "—" if v is None else f"{v / scale:.1f}"

lines = [
    "# perf_bench summary", "",
    "| arm | wall (s) | load_weight (s) | kv_cache (s) | scheduler_e2e (s) | request e2e (s) | retired dispatches | lane status | token gate (info) |",
    "|---|---|---|---|---|---|---|---|---|",
]
for arm, m in rows:
    gate = "PASS" if m.get("token_gate_correct") else "FAIL"
    if m.get("timed_out"):
        gate = "timeout"
    lines.append(
        f"| {arm} | {fmt(m,'wall_s')} | {fmt(m,'load_weight_s')} | "
        f"{fmt(m,'kv_cache_allocation_s')} | {fmt(m,'scheduler_e2e_s')} | "
        f"{fmt(m,'request_e2e_s')} | {m.get('retired_dispatches', 0)} | "
        f"{m.get('lane_status')} | {gate} |")
by = {a: m for a, m in rows}
if "legacy" in by and "full" in by and by["full"].get("wall_s"):
    lw, fw = by["legacy"].get("wall_s"), by["full"]["wall_s"]
    if lw:
        lines += ["", f"legacy → full wall speedup: **{lw / fw:.2f}×** "
                      f"(env-switchable optimizations on one binary)."]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
