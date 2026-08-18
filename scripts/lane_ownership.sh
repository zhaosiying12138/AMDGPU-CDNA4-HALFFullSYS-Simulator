#!/usr/bin/env bash
# Attribute managed gem5 simulators to the model lane that started them.
#
# This exists because a managed gem5 carries almost no identity. It is spawned
# with a scrubbed environment -- PATH, HOME, TMPDIR, XDG_CACHE_HOME, LC_ALL and
# nothing else -- so it cannot be tagged through the environment; and when the
# process that started it dies, it is reparented to init, so the process tree
# cannot answer the question either. Two real failures followed from guessing:
#
#   * Progress was computed by summing dispatch traces across every run
#     directory, which reported one lane at 92% when it was at 38% because
#     abandoned simulators from other lanes were counted into it.
#   * A cleanup matched processes with `grep -E 'qwen35_inference'` and killed
#     a healthy 500-dispatch lane along with its intended target.
#
# The fix is to give each lane its own run root through SAGR_MANAGED_RUN_ROOT
# (self-amdgpu-runtime/src/managed_session.c). Every run directory that lane
# creates then lives under a path only that lane uses, which makes both
# counting and killing exact.
#
# Roots live under /tmp rather than beside the lane log on purpose: the bridge
# endpoint is <run_dir>/bridge.sock and a AF_UNIX sun_path is capped at 108
# bytes, of which the run directory name and socket name already consume 43.
set -u

# The run root belonging to one lane. Short by construction.
lane_run_root() {  # lane_run_root <name>
  printf '/tmp/sagr-lane-%s' "$1"
}

# PIDs of every live gem5 this lane owns, including ones reparented to init.
lane_gem5_pids() {  # lane_gem5_pids <name>
  local name=$1 root pid outdir
  root=$(lane_run_root "$name")
  for pid in $(ps -eo pid,args --no-headers | awk '/[g]em5\.opt/{print $1}'); do
    outdir=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null \
             | grep -m1 -F -- "${root}/") || continue
    [[ -n $outdir ]] && printf '%s ' "$pid"
  done
}

# Retired dispatches for one lane, summed across the run directories under its
# own root only. Traces are counted whether or not their simulator is still
# alive, so a lane that finished still reports what it achieved.
lane_dispatch_total() {  # lane_dispatch_total <name>
  local name=$1 root total=0 n trace
  root=$(lane_run_root "$name")
  for trace in "${root}"/self-amdgpu-opencl-run.*/dispatch-trace.jsonl; do
    [[ -r $trace ]] || continue
    n=$(wc -l < "$trace" 2>/dev/null || echo 0)
    total=$((total + n))
  done
  printf '%s' "$total"
}

# Kill every simulator this lane owns. Safe to call while other lanes run: a
# process is killed only when its --outdir lies under this lane's root.
lane_reap_gem5() {  # lane_reap_gem5 <name>
  local name=$1 pid killed=0
  for pid in $(lane_gem5_pids "$name"); do
    kill -KILL "$pid" 2>/dev/null && killed=$((killed + 1))
  done
  printf '%s' "$killed"
}

# Completed wavefronts for one lane, from the most recent stats dump only.
#
# This is the liveness signal that dispatch retirement cannot provide. One
# kernel in this workload retires nothing for tens of minutes, so a watchdog
# keyed only on dispatches kills healthy runs.
#
# Two traps are baked into this function, both hit for real:
#
#  1. gem5 APPENDS to stats.txt on every SIGUSR1. Summing every line that
#     matches a counter therefore sums every historical dump, so the number
#     rises even when the simulator is completely frozen -- one lane was
#     reported as advancing through "102 billion instructions" while nothing
#     at all was happening. Only the last dump may be read.
#
#  2. Executed instructions are the wrong counter even when read correctly.
#     A wavefront spinning on `s_load_dword / s_cmp_eq_u32 / s_cbranch_scc0`
#     retires millions of scalar instructions per minute forever. Completed
#     wavefronts is the signal that actually distinguishes work from spinning:
#     it was frozen for over an hour on a lane whose instruction count was
#     climbing steadily.
#
# gem5 dumps stats on SIGUSR1 without dying (src/sim/init_signals.cc), so no
# debugger and no sudo is needed -- which matters because Yama ptrace_scope=1
# blocks an unprivileged gdb and the watchdog cannot authenticate.
lane_completed_wavefronts() {  # lane_completed_wavefronts <name>
  local name=$1 pid outdir total=0 value
  for pid in $(lane_gem5_pids "$name"); do
    kill -USR1 "$pid" 2>/dev/null
  done
  sleep 4
  for pid in $(lane_gem5_pids "$name"); do
    outdir=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null \
             | grep -m1 -oE '/tmp/sagr-lane-[^ ]*/m5out') || continue
    # Last occurrence wins: the file holds one block per dump.
    value=$(awk '/CUs\.completedWfs/{v=$2} END{printf "%d", v+0}' \
            "${outdir}/stats.txt" 2>/dev/null || echo 0)
    total=$((total + value))
  done
  printf '%s' "$total"
}

# Ask each of this lane's simulators to print a backtrace into its own gem5.log.
# gem5's SIGABRT handler prints the current tick plus a full backtrace, so a
# hang can be diagnosed without a debugger.
lane_dump_backtraces() {  # lane_dump_backtraces <name> <destination>
  local name=$1 destination=$2 pid outdir
  for pid in $(lane_gem5_pids "$name"); do
    kill -ABRT "$pid" 2>/dev/null
  done
  sleep 5
  for pid in $(lane_gem5_pids "$name"); do
    outdir=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null \
             | grep -m1 -oE '/tmp/sagr-lane-[^ ]*/m5out') || continue
    {
      echo "### pid ${pid}"
      tail -80 "${outdir%/m5out}/gem5.log" 2>/dev/null
    } >> "$destination" 2>&1
  done
}

# The first fatal error any of this lane's simulators reported, if any.
#
# A gem5 panic kills that simulator instantly, but the engine above it does not
# always notice: it can sit waiting on a queue that will never drain, so the
# lane looks "running" while nothing can ever happen again. Both TP2 lanes did
# exactly that -- every worker simulator had already aborted with
#
#   vopc.cc:5190: panic: SDWA not supported for v_cmp_gt_u32
#
# yet the lanes reported progress for another quarter of an hour, because the
# retired-dispatch count includes the traces those dead workers left behind.
# Detecting the panic directly turns a 45-minute watchdog timeout into a
# report within a minute, with the cause already in hand.
lane_fatal_error() {  # lane_fatal_error <name>
  local name=$1 root log
  root=$(lane_run_root "$name")
  for log in "${root}"/self-amdgpu-opencl-run.*/gem5.log; do
    [[ -r $log ]] || continue
    grep -m1 -hE '^[^ ]*: (panic|fatal):' "$log" 2>/dev/null && return 0
  done
  return 1
}
