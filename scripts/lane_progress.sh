#!/usr/bin/env bash
# Print model-lane progress as kernel launches completed against the expected
# total for one generated token.
#
# "Still running" is not a progress report. A lane can burn hours at 100% CPU
# having retired zero dispatches -- that has happened here -- so the only
# honest progress signal is retired kernel dispatches, which gem5 appends to
# dispatch-trace.jsonl as each one retires.
#
# Progress is reported PER LANE, never as one global number. Two earlier
# versions of this script summed across run directories and were both wrong:
# the first counted every historical directory in /tmp and reported 396% of a
# run that had barely started; the second counted only live simulators but
# still pooled unrelated lanes together, and reported 92% for a lane that was
# actually at 38% because abandoned simulators from other lanes were still
# alive and being counted. A lane is identified by its run root
# (SAGR_MANAGED_RUN_ROOT, recorded in status.json), which is the only durable
# identity a managed gem5 carries: its environment is scrubbed and it is
# reparented to init when its owner dies.
#
# The expected total is anchored on measurement, not architecture arithmetic:
# two independent Qwen3.5-0.8B TP1 runs reached 869 and 901 retired dispatches
# at the point they died, both past prefill. So roughly 900 dispatches carry
# this workload from model load through prefill to one emitted token. Override
# with SAGR_LANE_EXPECTED_DISPATCHES when the workload changes.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
expected=${SAGR_LANE_EXPECTED_DISPATCHES:-900}
sample_seconds=${SAGR_LANE_SAMPLE_SECONDS:-0}

# shellcheck source=scripts/lane_ownership.sh
source "${ROOT}/scripts/lane_ownership.sh"

bar() {  # bar <done> <total>
  local done=$1 total=$2 width=28 filled i
  (( total > 0 )) || total=1
  filled=$(( done * width / total ))
  (( filled > width )) && filled=$width
  (( filled < 0 )) && filled=0
  printf '['
  for (( i = 0; i < filled; i++ )); do printf '#'; done
  for (( i = filled; i < width; i++ )); do printf '.'; done
  printf ']'
}

lane_names=()
for lane in "$ROOT"/artifacts/lanes/*/; do
  [[ -d $lane ]] || continue
  lane_names+=("$(basename "$lane")")
done

if (( ${#lane_names[@]} == 0 )); then
  echo "no lanes under ${ROOT}/artifacts/lanes"
  exit 0
fi

# First sample.
declare -A first
for name in "${lane_names[@]}"; do
  first["$name"]=$(lane_dispatch_total "$name")
done

declare -A rate
if (( sample_seconds > 0 )); then
  sleep "$sample_seconds"
  for name in "${lane_names[@]}"; do
    second=$(lane_dispatch_total "$name")
    delta=$(( second - ${first["$name"]} ))
    remaining=$(( expected - second ))
    (( remaining < 0 )) && remaining=0
    rate["$name"]=$(python3 - "$delta" "$sample_seconds" "$remaining" <<'PY'
import sys
delta, seconds, remaining = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rate = delta / (seconds / 60) if seconds else 0.0
eta = "n/a" if rate <= 0 else f"{remaining / rate:.0f}m"
print(f"{rate:.1f}/min eta={eta}")
PY
)
    first["$name"]=$second
  done
fi

for name in "${lane_names[@]}"; do
  lane="${ROOT}/artifacts/lanes/${name}/"
  count=${first["$name"]}
  percent=$(( expected > 0 ? count * 100 / expected : 0 ))
  state="unknown"; detail=""
  if [[ -r "${lane}status.json" ]]; then
    read -r state detail < <(python3 - "${lane}status.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("state", "unknown"), d.get("detail", ""))
except Exception:
    print("unknown", "")
PY
)
  fi
  live=$(lane_gem5_pids "$name" | wc -w)
  printf '%-18s %s %4s/%-4s (%3s%%)  %-9s gem5=%-2s %s\n' \
    "$name" "$(bar "$count" "$expected")" "$count" "$expected" "$percent" \
    "$state" "$live" "${rate[$name]:-}"
  if [[ -r "${lane}run.log" ]]; then
    phase=$(grep -v '\[\*\*\*rocr\*\*\*\]' "${lane}run.log" 2>/dev/null | tail -1 | cut -c1-96)
    [[ -n $phase ]] && printf '                   %s\n' "$phase"
  fi
done
