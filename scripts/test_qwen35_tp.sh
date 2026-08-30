#!/usr/bin/env bash
# Reproduce the accepted Qwen3.5 TP lanes with isolated state and fail-closed
# validation. Heavy model cases are always explicit and run serially.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
RUNNER="${ROOT}/scripts/run_engine_lane.sh"
GEM5_ROOT=${SAGR_TEST_GEM5_ROOT:-/home/zhaosiying/zcode-gem5-hybrid2}
GEM5=${SAGR_TEST_GEM5:-${GEM5_ROOT}/build/VEGA_X86/gem5.opt.hybridfastwrap2}
GEM5_CONFIG=${SAGR_TEST_GEM5_CONFIG:-${GEM5_ROOT}/configs/example/gemsim/host_dispatch.py}
OUTPUT_BASE=${SAGR_TEST_OUTPUT_BASE:-${ROOT}/artifacts/qwen35-selftest}
RUNROOT_BASE=${SAGR_TEST_RUNROOT_BASE:-/home/zhaosiying}
TOKENS=10
PROMPT=${SAGR_TEST_PROMPT:-'为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？'}
TIMEOUT_OVERRIDE=
ALLOW_BUSY=0
CURRENT_RUN_ROOT=

usage() {
  cat <<'EOF'
Usage:
  scripts/test_qwen35_tp.sh 0.8b-tp2 [options]
  scripts/test_qwen35_tp.sh 9b-tp4   [options]
  scripts/test_qwen35_tp.sh all      [options]

Options:
  --tokens N              Generate and compare 1..10 tokens (default: 10).
  --prompt TEXT           Use the pinned text prompt (must match its golden).
  --timeout-seconds N     Override the case timeout.
  --allow-busy-host       Run even when another model lane/gem5 is active.
  -h, --help              Show this help without starting a model.

Defaults:
  0.8B TP2 timeout: 10800 seconds (3 hours)
  9B TP4 timeout:   43200 seconds (12 hours)

Every case gets a private managed run root, Triton cache, AITER configuration,
lane log, and JSON report. A PASS requires an exact checkpoint token match,
runner status 0, no simulator panic/fatal/vector-access error, no NCCL watchdog
or HIP 209 error, and no process left behind by that run.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

is_positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

owned_pids() { # owned_pids <run-root>
  local run_root=$1 proc pid cmdline entry owns_env
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    [[ $pid != $$ && $pid != $BASHPID ]] || continue
    cmdline=$(tr '\0' ' ' <"${proc}/cmdline" 2>/dev/null || true)
    if [[ $cmdline == *"${run_root}/"* ]]; then
      printf '%s\n' "$pid"
      continue
    fi
    owns_env=0
    while IFS= read -r -d '' entry; do
      if [[ $entry == "SAGR_MANAGED_RUN_ROOT=${run_root}" ]]; then
        owns_env=1
        break
      fi
    done <"${proc}/environ" 2>/dev/null || true
    if ((owns_env)); then
      printf '%s\n' "$pid"
    fi
  done | sort -nu
}

reap_owned() { # reap_owned <run-root>
  local run_root=$1 pids pid
  mapfile -t pids < <(owned_pids "$run_root")
  ((${#pids[@]})) || return 0
  printf 'Cleaning only processes owned by %s: %s\n' \
    "$run_root" "${pids[*]}" >&2
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 2
  mapfile -t pids < <(owned_pids "$run_root")
  for pid in "${pids[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
}

on_signal() {
  local signal=$1
  if [[ -n $CURRENT_RUN_ROOT ]]; then
    reap_owned "$CURRENT_RUN_ROOT"
  fi
  printf 'Interrupted by %s\n' "$signal" >&2
  exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

MODE=${1:-}
case "$MODE" in
  -h|--help) usage; exit 0 ;;
  0.8b-tp2|9b-tp4|all) shift ;;
  '') usage; exit 2 ;;
  *) die "unknown case: ${MODE}" ;;
esac

while (($#)); do
  case "$1" in
    --tokens) TOKENS=${2:-}; shift 2 ;;
    --prompt) PROMPT=${2:-}; shift 2 ;;
    --timeout-seconds) TIMEOUT_OVERRIDE=${2:-}; shift 2 ;;
    --allow-busy-host) ALLOW_BUSY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

is_positive_integer "$TOKENS" || die '--tokens must be a positive integer'
((TOKENS <= 10)) || die '--tokens cannot exceed the frozen 10-token oracle'
if [[ -n $TIMEOUT_OVERRIDE ]]; then
  is_positive_integer "$TIMEOUT_OVERRIDE" ||
    die '--timeout-seconds must be a positive integer'
fi
[[ -x $RUNNER ]] || die "runner is not executable: ${RUNNER}"
[[ -x $GEM5 ]] || die "gem5 wrapper is not executable: ${GEM5}"
[[ -r $GEM5_CONFIG ]] || die "gem5 config is not readable: ${GEM5_CONFIG}"
mkdir -p "$OUTPUT_BASE"

# Prevent two invocations of this acceptance script from sharing host capacity.
exec 9>"${OUTPUT_BASE}/.lock"
flock -n 9 || die "another qwen35 self-test holds ${OUTPUT_BASE}/.lock"

if (( ! ALLOW_BUSY )); then
  busy=$(ps -eo pid=,args= | awk '
    /[q]wen35_inference\.py/ || /[g]em5\.opt/ {print}
  ')
  if [[ -n $busy ]]; then
    printf '%s\n' "$busy" >&2
    die 'another Qwen/gem5 workload is active; wait or pass --allow-busy-host'
  fi
fi

run_case() { # run_case <case-name>
  local case_name=$1 short tp model timeout_seconds stamp artifact_dir
  local log report started_ns ended_ns wall_ms runner_status report_status
  local -a leftover

  case "$case_name" in
    0.8b-tp2)
      short=08tp2
      tp=2
      model="${ROOT}/models/Qwen3.5-0.8B"
      timeout_seconds=${TIMEOUT_OVERRIDE:-10800}
      ;;
    9b-tp4)
      short=9tp4
      tp=4
      model="${ROOT}/models/Qwen3.5-9B"
      timeout_seconds=${TIMEOUT_OVERRIDE:-43200}
      ;;
    *) die "internal unknown case: ${case_name}" ;;
  esac

  [[ -d $model ]] || die "model is absent: ${model}"
  stamp=$(date +%Y%m%dT%H%M%S)
  artifact_dir="${OUTPUT_BASE}/${stamp}-${short}-${TOKENS}tok"
  mkdir -p "$artifact_dir"
  log="${artifact_dir}/lane.log"
  report="${artifact_dir}/report.json"
  # Keep this short: managed bridge endpoints are AF_UNIX paths.
  CURRENT_RUN_ROOT="${RUNROOT_BASE}/zqt-${short}-${stamp}-$$"
  mkdir -p "$CURRENT_RUN_ROOT"

  printf '\n[%s] case=%s tokens=%s timeout=%ss\n' \
    "$(date -Is)" "$case_name" "$TOKENS" "$timeout_seconds"
  printf 'lane log: %s\nrun root: %s\n' "$log" "$CURRENT_RUN_ROOT"
  printf 'Progress: tail -f %q\n' "$log"

  started_ns=$(date +%s%N)
  set +e
  SAGR_LANE_CACHE_ROOT="${CURRENT_RUN_ROOT}/cache" \
  SAGR_MANAGED_RUN_ROOT="$CURRENT_RUN_ROOT" \
  SAGR_MANAGED_GEM5="$GEM5" \
  SAGR_MANAGED_GEM5_CONFIG="$GEM5_CONFIG" \
  SAGR_SGLANG_USE_AITER=1 \
    timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" \
      "$RUNNER" --engine sglang --tp "$tp" --model "$model" \
      --max-new-tokens "$TOKENS" --prompt "$PROMPT" "$log"
  runner_status=$?
  set -e
  ended_ns=$(date +%s%N)
  wall_ms=$(((ended_ns - started_ns) / 1000000))

  # A clean engine shutdown should leave nothing. Residuals make the run fail,
  # then are removed by exact run-root ownership so later tests are not tainted.
  mapfile -t leftover < <(owned_pids "$CURRENT_RUN_ROOT")
  if ((${#leftover[@]})); then
    printf 'Residual owned PIDs: %s\n' "${leftover[*]}" >&2
    reap_owned "$CURRENT_RUN_ROOT"
  fi

  set +e
  python3 - "$case_name" "$TOKENS" "$runner_status" "$wall_ms" \
    "$log" "$CURRENT_RUN_ROOT" "$report" "${leftover[*]:-}" <<'PY'
import json
from pathlib import Path
import re
import sys

case_name, tokens_s, status_s, wall_ms_s, log_s, run_root_s, report_s, leftovers = sys.argv[1:]
tokens = int(tokens_s)
runner_status = int(status_s)
wall_seconds = int(wall_ms_s) / 1000.0
log_path = Path(log_s)
run_root = Path(run_root_s)
text = log_path.read_text(errors="replace") if log_path.is_file() else ""

gate = None
for line in text.splitlines():
    marker = "token_golden="
    if marker in line:
        try:
            gate = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            gate = None

output = None
decoder = json.JSONDecoder()
for line in text.splitlines():
    start = line.find('{"output_ids":')
    if start >= 0:
        try:
            candidate, _ = decoder.raw_decode(line[start:])
            output = candidate
        except json.JSONDecodeError:
            pass

gem5_logs = sorted(run_root.glob("self-amdgpu-opencl-run.*/gem5.log"))
gem5_failures = []
gem5_pattern = re.compile(
    r"\b(?:panic|fatal)\b|host-native vector memory access failed",
    re.IGNORECASE,
)
for path in gem5_logs:
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if gem5_pattern.search(line):
            gem5_failures.append(f"{path}:{number}:{line[:300]}")

lane_failure_patterns = {
    "nccl_watchdog": re.compile(r"ProcessGroupNCCL watchdog got stuck", re.I),
    "hip_209": re.compile(r"(?:HIP.*Code:\s*209|no kernel image is available)", re.I),
}
lane_failures = [name for name, pattern in lane_failure_patterns.items() if pattern.search(text)]
finished_zero = bool(re.search(r"^finished=.* status=0$", text, re.MULTILINE))
actual = gate.get("actual_token_ids") if isinstance(gate, dict) else None
gate_passed = bool(
    isinstance(gate, dict)
    and gate.get("correct") is True
    and gate.get("checkpoint_weights") is True
    and isinstance(actual, list)
    and len(actual) == tokens
)
e2e_seconds = None
if isinstance(output, dict):
    value = output.get("meta_info", {}).get("e2e_latency")
    if isinstance(value, (int, float)):
        e2e_seconds = float(value)

failures = []
if runner_status != 0:
    failures.append(f"runner_status={runner_status}")
if not finished_zero:
    failures.append("missing finished status=0")
if not gate_passed:
    failures.append("checkpoint token gate did not pass")
if not gem5_logs:
    failures.append("no gem5 logs found")
if gem5_failures:
    failures.append(f"gem5 failures={len(gem5_failures)}")
if lane_failures:
    failures.extend(lane_failures)
if leftovers.strip():
    failures.append(f"residual_pids={leftovers.strip()}")

report = {
    "schema": "amdgpu-sim.qwen35-tp-selftest.v1",
    "case": case_name,
    "passed": not failures,
    "tokens": tokens,
    "runner_status": runner_status,
    "finished_status_zero": finished_zero,
    "wall_seconds": wall_seconds,
    "average_wall_seconds_per_token": wall_seconds / tokens,
    "request_e2e_seconds": e2e_seconds,
    "average_request_seconds_per_token": (
        e2e_seconds / tokens if e2e_seconds is not None else None
    ),
    "actual_token_ids": actual,
    "token_gate": gate,
    "gem5_logs_scanned": len(gem5_logs),
    "gem5_failures": gem5_failures,
    "lane_failures": lane_failures,
    "residual_pids": [int(pid) for pid in leftovers.split()] if leftovers.strip() else [],
    "failures": failures,
    "lane_log": str(log_path),
    "run_root": str(run_root),
}
Path(report_s).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

print(f"result={'PASS' if report['passed'] else 'FAIL'}")
print(f"tokens={actual}")
print(f"wall={wall_seconds:.3f}s average_wall={wall_seconds / tokens:.3f}s/token")
if e2e_seconds is not None:
    print(f"request_e2e={e2e_seconds:.3f}s average_request={e2e_seconds / tokens:.3f}s/token")
print(f"gem5_logs={len(gem5_logs)} gem5_failures={len(gem5_failures)}")
print(f"report={report_s}")
if failures:
    print("failures=" + "; ".join(failures))
raise SystemExit(0 if report["passed"] else 1)
PY
  report_status=$?
  set -e
  CURRENT_RUN_ROOT=
  return "$report_status"
}

case "$MODE" in
  0.8b-tp2|9b-tp4) run_case "$MODE" ;;
  all)
    run_case 0.8b-tp2
    run_case 9b-tp4
    ;;
esac

printf '\nAll requested Qwen3.5 TP self-tests passed.\n'
