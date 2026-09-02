#!/usr/bin/env bash
# Start / stop / restart / status for a standalone gem5 simulator fleet.
#
# gem5-session start [N]     boot N gem5 instances (default 1), each with
#                            its own 0700 session dir, bridge endpoint, and
#                            SMI registry lease so rocm-smi shows them ON
# gem5-session stop [N|--all]
#                            stop instance set (default: all running)
# gem5-session restart [N]   stop all, then boot N (default: last N)
# gem5-session status        per-instance pid / endpoint / dispatches
#
# Options: --accurate  use the timing-accurate binary instead of fastwrap.
# The environment's activation sources session.env after start, so new
# shells (or re-activation) attach to instance 0 automatically.
set -u

ROOT="/home/zhaosiying/zcode-lane"
GEM5_ACCURATE="${ROOT}/projects/gem5/build/VEGA_X86/gem5.opt"
GEM5_FAST="${ROOT}/projects/gem5/build/VEGA_X86/gem5.opt.fastwrap"
GEM5_CONFIG="${ROOT}/projects/gem5/configs/example/gemsim/host_dispatch.py"
SMI_PUBLISH="${ROOT}/tools/gemsim_smi_publish.py"
PYTHON="${ROOT}/env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d/bin/python"

SESSION_ROOT="${SAGR_TOOLS_SESSION_DIR:-/tmp/amdgpu-sim-tools-session}"
SESSION_ENV="${SAGR_TOOLS_SESSION_ENV:-${SESSION_ROOT}/session.env}"
RUNTIME_BUILD="${ROOT}/projects/self-amdgpu-runtime/build/cp28-runtime-clang"
ROCR_LIB="${SAGR_ROCR_LIBRARY_DIR:-${ROOT}/build/rocr-stage-zcode/lib}"

cmd="${1:-status}"; shift 2>/dev/null || true
accurate=0
count=""
for arg in "$@"; do
  case $arg in
    --accurate) accurate=1 ;;
    --all) count="all" ;;
    ''|*[!0-9]*) : ;;
    *) count=$arg ;;
  esac
done

_instances_running() {
  local found=0 inst dir
  for inst in $(seq 0 15); do
    dir="${SESSION_ROOT}/instance-${inst}"
    if [[ -f ${dir}/gem5.pid ]] && kill -0 "$(cat "${dir}/gem5.pid")" 2>/dev/null; then
      found=$((found + 1))
    fi
  done
  echo $found
}

_stop_one() {
  local inst=$1 dir pid pgid hpid
  dir="${SESSION_ROOT}/instance-${inst}"
  if [[ -f ${dir}/gem5.pid ]] && kill -0 "$(cat "${dir}/gem5.pid")" 2>/dev/null; then
    pid=$(cat "${dir}/gem5.pid")
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [[ -f ${dir}/smi-holder.pid ]] && { hpid=$(cat "${dir}/smi-holder.pid"); kill -TERM "$hpid" 2>/dev/null; }
    kill -TERM -- -"$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    sleep 2
    kill -KILL -- -"$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    echo "instance ${inst}: stopped (was pid ${pid})"
  fi
  rm -f "${dir}/gem5.pid" "${dir}/smi-holder.pid" \
        "/tmp/amdgpu-sim-smi-$(id -u)/device-$(printf '%02d' "${inst}").bin" 2>/dev/null
}

_do_start() {
  local n=${count:-1}
  [[ $n =~ ^[0-9]+$ ]] || { echo "instance count must be a number" >&2; return 2; }
  (( n >= 1 && n <= 16 )) || { echo "instance count must be 1..16" >&2; return 2; }
  local binary=$GEM5_FAST
  (( accurate )) && binary=$GEM5_ACCURATE
  for f in "$binary" "$GEM5_CONFIG" "${RUNTIME_BUILD}/libself_amdgpu_runtime.so.1" \
           "${ROCR_LIB}/libhsa-runtime64.so.1" "$SMI_PUBLISH" "$PYTHON"; do
    [[ -f $f ]] || { echo "missing dependency: $f" >&2; return 2; }
  done
  # A same-session restart reuses the fleet identity so world_size stays
  # consistent across restarts of the same logical job.
  local job_uuid
  if [[ -f ${SESSION_ROOT}/job-uuid ]]; then
    job_uuid=$(cat "${SESSION_ROOT}/job-uuid")
  else
    job_uuid=$(cat /proc/sys/kernel/random/uuid | tr -d '-')
  fi

  local inst dir endpoint pid i
  for inst in $(seq 0 $((n - 1))); do
    dir="${SESSION_ROOT}/instance-${inst}"
    if [[ -f ${dir}/gem5.pid ]] && kill -0 "$(cat "${dir}/gem5.pid")" 2>/dev/null; then
      echo "instance ${inst}: already running (pid $(cat "${dir}/gem5.pid"))"
      continue
    fi
    rm -rf "$dir"
    mkdir -m 700 -p "$dir"
    chmod 700 "$dir"
    mkdir -m 700 -p "${dir}/m5out"
    endpoint="${dir}/bridge.sock"
    (
      cd "$ROOT"
      setsid nohup env \
        LD_LIBRARY_PATH="${RUNTIME_BUILD}:${ROCR_LIB}" \
        HSA_ENABLE_DXG_DETECTION=0 \
        HSA_ENABLE_INTERRUPT=0 \
        "$binary" \
        --listener-mode=on \
        --outdir "${dir}/m5out" \
        "$GEM5_CONFIG" \
        --endpoint "$endpoint" \
        --dispatch-trace-path "${dir}/dispatch-trace.jsonl" \
        --epoch 1 --job-uuid "$job_uuid" \
        --rank "$inst" --world-size "$n" \
        --startup-timeout-ms 86400000 \
        --handshake-timeout-ms 86400000 \
        --run-timeout-ms 86400000 \
        >"${dir}/gem5.log" 2>&1 &
      echo $! > "${dir}/gem5.pid"
    )
    pid=$(cat "${dir}/gem5.pid")
    for i in $(seq 1 120); do
      [[ -S $endpoint ]] && break
      kill -0 "$pid" 2>/dev/null || { echo "instance ${inst}: gem5 died during startup (log: ${dir}/gem5.log)" >&2; break; }
      sleep 1
    done
    if [[ ! -S $endpoint ]]; then
      echo "instance ${inst}: endpoint never appeared" >&2
      continue
    fi
    chmod 600 "$endpoint"
    (
      setsid nohup "$PYTHON" "$SMI_PUBLISH" \
        --slot "$inst" --daemon-pid "$pid" \
        --rank "$inst" --world "$n" \
        --endpoint "$endpoint" --job-uuid "$job_uuid" \
        >"${dir}/smi-holder.log" 2>&1 &
      echo $! > "${dir}/smi-holder.pid"
    )
    echo "instance ${inst}: started pid ${pid} endpoint ${endpoint}"
  done

  mkdir -m 700 -p "$SESSION_ROOT" 2>/dev/null || chmod 700 "$SESSION_ROOT"
  echo "$job_uuid" > "${SESSION_ROOT}/job-uuid"
  echo "$n" > "${SESSION_ROOT}/instance-count"
  cat > "$SESSION_ENV" <<ENVEOF
export SAGR_GENERIC_BRIDGE_ENDPOINT="${SESSION_ROOT}/instance-0/bridge.sock"
export SAGR_TOOLS_INSTANCE_COUNT="${n}"
export SAGR_TOOLS_INSTANCE_ENDPOINTS="$(for inst in $(seq 0 $((n - 1))); do printf '%s ' "${SESSION_ROOT}/instance-${inst}/bridge.sock"; done)"
ENVEOF
  chmod 600 "$SESSION_ENV"
  echo "fleet: $(_instances_running) instance(s) running; attach endpoint = instance-0"
  echo "  new shells pick it up automatically; now: source ${SESSION_ENV}"
}

_do_stop() {
  local inst pattern=${count:-all}
  if [[ $pattern == all ]]; then
    for inst in $(seq 0 15); do _stop_one "$inst"; done
    rm -f "$SESSION_ENV" "${SESSION_ROOT}/job-uuid" "${SESSION_ROOT}/instance-count"
  else
    _stop_one "$pattern"
  fi
}

_do_status() {
  local inst dir pid dispatches running=0
  for inst in $(seq 0 15); do
    dir="${SESSION_ROOT}/instance-${inst}"
    if [[ -f ${dir}/gem5.pid ]] && kill -0 "$(cat "${dir}/gem5.pid")" 2>/dev/null; then
      pid=$(cat "${dir}/gem5.pid")
      dispatches=0
      [[ -f ${dir}/dispatch-trace.jsonl ]] && dispatches=$(wc -l < "${dir}/dispatch-trace.jsonl")
      echo "instance ${inst}: RUNNING pid ${pid} dispatches ${dispatches}"
      echo "  endpoint: ${dir}/bridge.sock"
      running=$((running + 1))
    fi
  done
  if (( running == 0 )); then
    echo "gem5 fleet STOPPED"
  else
    echo "fleet: ${running} instance(s) running; attach: source ${SESSION_ENV}"
  fi
}

case $cmd in
  start)   _do_start ;;
  stop)    _do_stop ;;
  restart) _do_stop; sleep 1
           if [[ -z $count && -f ${SESSION_ROOT}/instance-count ]]; then
             count=$(cat "${SESSION_ROOT}/instance-count")
           fi
           _do_start ;;
  status)  _do_status ;;
  *) echo "usage: gem5-session start|stop|restart|status [N|--all] [--accurate]" >&2; exit 2 ;;
esac
