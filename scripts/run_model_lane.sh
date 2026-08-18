#!/usr/bin/env bash
# Run one model lane fully detached, with a stall watchdog.
#
# The lane must survive the operator's session ending: a dropped network, a
# closed terminal, or an assistant hitting a usage limit must not stop or
# orphan a 50-70 minute run. The lane therefore runs under setsid in its own
# session, writes an append-only log plus a machine-readable status file, and
# is supervised by a watchdog.
#
# The watchdog exists because of a real failure: a gem5 defect once left the
# simulator spinning at 100% CPU for five and a half hours having retired zero
# dispatches. Wall-clock alone cannot distinguish that from a legitimately slow
# kernel, but "no dispatch retired for N minutes while gem5 burns CPU" can.
#
# Every simulator this lane starts is confined to its own run root, so counting
# and killing are exact even when several lanes run at once and even after a
# simulator is reparented to init. See scripts/lane_ownership.sh for why that
# matters -- an earlier version pooled all lanes together and both misreported
# progress and killed a healthy lane.
#
#   scripts/run_model_lane.sh --name sglang-tp1 --runner /tmp/run_sglang_tp1_head.sh
#
# Status lands in artifacts/lanes/<name>/status.json and is safe to poll.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
# shellcheck source=scripts/lane_ownership.sh
source "${ROOT}/scripts/lane_ownership.sh"

name=""
runner=""
# Two thresholds, because dispatch retirement alone cannot tell a slow kernel
# from a hung one. At --warn-minutes without a retired dispatch the lane is
# probed for executed instructions; if those are advancing it is inside a long
# kernel and is left alone. Only when instructions stop moving for
# --stall-minutes is it killed. A single-threshold watchdog killed a healthy
# vLLM lane at 40 dispatches while its simulator was executing normally.
warn_minutes=10
stall_minutes=45
max_hours=12

while (($#)); do
  case "$1" in
    --name) name=${2:?}; shift 2 ;;
    --runner) runner=${2:?}; shift 2 ;;
    --warn-minutes) warn_minutes=${2:?}; shift 2 ;;
    --stall-minutes) stall_minutes=${2:?}; shift 2 ;;
    --max-hours) max_hours=${2:?}; shift 2 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -z $name || -z $runner ]]; then
  printf 'usage: %s --name NAME --runner PATH [--stall-minutes N] [--max-hours N]\n' "$0" >&2
  exit 2
fi
if [[ ! -r $runner ]]; then
  printf 'runner is not readable: %s\n' "$runner" >&2
  exit 2
fi

lane_dir="${ROOT}/artifacts/lanes/${name}"
mkdir -p "$lane_dir"
log="${lane_dir}/run.log"
status="${lane_dir}/status.json"
run_root=$(lane_run_root "$name")

write_status() {  # write_status <state> <detail>
  local tmp="${status}.tmp"
  printf '{"schema":"amdgpu-sim.model-lane.v2","name":"%s","state":"%s","detail":"%s","pid":%s,"run_root":"%s","log":"%s","updated":"%s"}\n' \
    "$name" "$1" "$2" "${lane_pid:-0}" "$run_root" "$log" "$(date -Is)" > "$tmp"
  mv -f "$tmp" "$status"
}

# A previous incarnation of this lane may have left simulators behind: they are
# reparented to init and keep burning a core each. Nine such orphans were once
# found spinning at 80% CPU apiece, starving the one lane that was real. They
# are unambiguously this lane's because they sit under this lane's run root.
reaped=$(lane_reap_gem5 "$name")
rm -rf "${run_root:?}"
mkdir -p "$run_root"

lane_pid=0
write_status starting "reaped ${reaped} stale simulators; launching runner"

# The runner inherits the run root, so every managed session it starts -- and
# every session started by processes it forks, such as SGLang's scheduler and
# detokenizer workers -- lands under it.
SAGR_MANAGED_RUN_ROOT="$run_root" \
  setsid nohup bash "$runner" "$log" >"${lane_dir}/wrapper.out" 2>&1 &
lane_pid=$!
write_status running "runner started"

deadline=$(( $(date +%s) + max_hours * 3600 ))
last_dispatch=$(lane_dispatch_total "$name")
last_progress=$(date +%s)
last_instructions=0
last_instruction_progress=$(date +%s)

while kill -0 "$lane_pid" 2>/dev/null; do
  sleep 60
  now=$(date +%s)
  current=$(lane_dispatch_total "$name")
  if [[ $current -ne $last_dispatch ]]; then
    last_dispatch=$current
    last_progress=$now
    last_instruction_progress=$now
    write_status running "dispatches=${current}"
  elif (( now - last_progress > warn_minutes * 60 )); then
    # No dispatch has retired for a while. Before concluding anything, ask the
    # simulators whether they are still executing. This is the difference
    # between the bottleneck kernel -- which retires nothing for tens of
    # minutes by design -- and a genuine hang.
    instructions=$(lane_completed_wavefronts "$name")
    if (( instructions > last_instructions )); then
      last_instructions=$instructions
      last_instruction_progress=$now
      write_status running \
        "dispatches=${current} inside long kernel; completed_wavefronts=${instructions}"
    fi
  fi
  # A simulator panic is terminal for the lane even when the engine above it
  # has not noticed yet. Stop immediately with the cause rather than waiting
  # out the stall timeout.
  if fatal=$(lane_fatal_error "$name"); then
    capture="${lane_dir}/fatal-$(date +%Y%m%dT%H%M%S)"
    mkdir -p "$capture"
    {
      echo "# simulator fatal error"
      echo "lane=${name}"
      echo "dispatches=${current}"
      echo "error=${fatal}"
      date -Is
    } > "${capture}/summary.txt"
    for gem5_log in "${run_root}"/self-amdgpu-opencl-run.*/gem5.log; do
      [[ -r $gem5_log ]] || continue
      { echo "### ${gem5_log}"; tail -60 "$gem5_log"; } >> "${capture}/gem5-logs.txt"
    done
    tail -200 "$log" > "${capture}/run-log-tail.txt" 2>/dev/null
    write_status fatal "${fatal}; capture=${capture}"
    pkill -KILL -s "$lane_pid" 2>/dev/null || kill -KILL "$lane_pid" 2>/dev/null
    lane_reap_gem5 "$name" >/dev/null
    exit 5
  fi
  if (( now - last_instruction_progress > stall_minutes * 60 )); then
    # Capture the evidence *before* killing anything. A stall that is only
    # reported as "stalled" forces a full re-run to diagnose; a stall that
    # arrives with a backtrace is usually diagnosable on the spot. A 5.5 hour
    # window was once lost to a simulator hang that a single backtrace would
    # have identified immediately.
    capture="${lane_dir}/stall-$(date +%Y%m%dT%H%M%S)"
    mkdir -p "$capture"
    lane_pids=$(lane_gem5_pids "$name")
    {
      echo "# stall capture"
      echo "lane=${name}"
      echo "run_root=${run_root}"
      echo "dispatches=${current}"
      echo "completed_wavefronts=${last_instructions}"
      echo "dispatch_stalled_seconds=$(( now - last_progress ))"
      echo "instruction_stalled_seconds=$(( now - last_instruction_progress ))"
      date -Is
      echo
      echo "## this lane's gem5 processes"
      for pid in $lane_pids; do
        ps -o pid,ppid,etime,pcpu,rss,args --no-headers -p "$pid" 2>/dev/null
        grep -E '^(State|Threads|VmRSS|voluntary_ctxt)' "/proc/${pid}/status" 2>/dev/null
      done
    } > "${capture}/summary.txt" 2>&1
    # gem5 prints its own backtrace on SIGABRT, so no debugger is needed. That
    # matters: Yama ptrace_scope=1 blocks an unprivileged gdb here, and an
    # earlier capture produced an empty backtrace file for exactly that reason,
    # leaving a stall undiagnosable and forcing a re-run.
    lane_dump_backtraces "$name" "${capture}/backtraces.txt"
    for trace in "${run_root}"/self-amdgpu-opencl-run.*/dispatch-trace.jsonl; do
      [[ -r $trace ]] || continue
      tail -20 "$trace" >> "${capture}/trace-tail.jsonl" 2>/dev/null
    done
    tail -200 "$log" > "${capture}/run-log-tail.txt" 2>/dev/null

    write_status stalled "no completed wavefront for ${stall_minutes}m at dispatches=${current}; capture=${capture}"
    pkill -KILL -s "$lane_pid" 2>/dev/null || kill -KILL "$lane_pid" 2>/dev/null
    lane_reap_gem5 "$name" >/dev/null
    exit 3
  fi
  if (( now > deadline )); then
    write_status timeout "exceeded ${max_hours}h at dispatches=${current}"
    pkill -KILL -s "$lane_pid" 2>/dev/null || kill -KILL "$lane_pid" 2>/dev/null
    lane_reap_gem5 "$name" >/dev/null
    exit 4
  fi
done

wait "$lane_pid"
rc=$?
final=$(lane_dispatch_total "$name")
# The runner has exited; anything of this lane's still alive is an orphan.
lane_reap_gem5 "$name" >/dev/null
if (( rc == 0 )); then
  write_status completed "exit=0 dispatches=${final}"
else
  write_status failed "exit=${rc} dispatches=${final}"
fi
exit "$rc"
