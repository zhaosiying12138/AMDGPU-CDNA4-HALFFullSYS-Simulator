#!/usr/bin/env bash
# Start / stop / restart / status for ONE standalone gem5 simulator instance.
#
# The instance runs detached under its own session directory with a
# listener-mode bridge; programs in the tools environment attach to it
# through SAGR_GENERIC_BRIDGE_ENDPOINT (exported by the environment's
# activation from the session file this script writes).
#
# Usage: gem5-session start|stop|restart|status [--accurate]
#   start     boot one simulator (functional-fast by default; --accurate
#             for the timing-accurate binary)
#   stop      terminate the instance and clean its session dir
#   restart   stop then start
#   status    show pid, bridge socket, session dir, dispatch count
set -u

ROOT="/home/zhaosiying/zcode-lane"
GEM5_ACCURATE="${ROOT}/projects/gem5/build/VEGA_X86/gem5.opt"
GEM5_FAST="${ROOT}/projects/gem5/build/VEGA_X86/gem5.opt.fastwrap"
GEM5_CONFIG="${ROOT}/projects/gem5/configs/example/gemsim/host_dispatch.py"

SESSION_ROOT="${SAGR_TOOLS_SESSION_DIR:-/tmp/amdgpu-sim-tools-session}"
ENDPOINT="${SESSION_ROOT}/bridge.sock"
PIDFILE="${SESSION_ROOT}/gem5.pid"
LOGFILE="${SESSION_ROOT}/gem5.log"
SESSION_ENV="${SAGR_TOOLS_SESSION_ENV:-${SESSION_ROOT}/session.env}"
RUNTIME_BUILD="${ROOT}/projects/self-amdgpu-runtime/build/cp28-runtime-clang"
ROCR_LIB="${SAGR_ROCR_LIBRARY_DIR:-${ROOT}/build/rocr-stage-zcode/lib}"

cmd="${1:-status}"
[[ $# -gt 1 ]] && shift
accurate=0
for arg in "$@"; do
  [[ $arg == "--accurate" ]] && accurate=1
done

_instance_alive() {
  [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

_do_start() {
  if _instance_alive; then
    echo "gem5 already running (pid $(cat "$PIDFILE"), endpoint ${ENDPOINT})"
    return 0
  fi
  local binary=$GEM5_FAST
  (( accurate )) && binary=$GEM5_ACCURATE
  for f in "$binary" "$GEM5_CONFIG" "${RUNTIME_BUILD}/libself_amdgpu_runtime.so.1" \
           "${ROCR_LIB}/libhsa-runtime64.so.1"; do
    [[ -f $f ]] || { echo "missing dependency: $f" >&2; return 2; }
  done
  rm -rf "$SESSION_ROOT"
  # gem5's evidence files require a 0700 owner-only parent (no group/other
  # bits), exactly like the lane run roots.
  mkdir -m 700 -p "$SESSION_ROOT"
  chmod 700 "$SESSION_ROOT"
  mkdir -m 700 -p "${SESSION_ROOT}/m5out"
  (
    cd "$ROOT"
    setsid nohup env \
      LD_LIBRARY_PATH="${RUNTIME_BUILD}:${ROCR_LIB}" \
      HSA_MODEL_TOPOLOGY="${SAGR_TOOLS_TOPOLOGY:-}" \
      "$binary" \
      --listener-mode=on \
      --outdir "${SESSION_ROOT}/m5out" \
      "$GEM5_CONFIG" \
      --endpoint "$ENDPOINT" \
      --dispatch-trace-path "${SESSION_ROOT}/dispatch-trace.jsonl" \
      --epoch 1 --job-uuid "$(cat /proc/sys/kernel/random/uuid)" \
      --rank 0 --world-size 1 \
      --startup-timeout-ms 86400000 \
      --handshake-timeout-ms 86400000 \
      --run-timeout-ms 86400000 \
      >"$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
  )
  for _ in $(seq 1 120); do
    [[ -S $ENDPOINT ]] && break
    _instance_alive || { echo "gem5 died during startup; log: $LOGFILE" >&2; tail -5 "$LOGFILE" >&2; return 1; }
    sleep 1
  done
  [[ -S $ENDPOINT ]] || { echo "endpoint never appeared: $ENDPOINT" >&2; return 1; }
  cat > "$SESSION_ENV" <<ENVEOF
export SAGR_GENERIC_BRIDGE_ENDPOINT="${ENDPOINT}"
ENVEOF
  echo "gem5 started: pid $(cat "$PIDFILE")"
  echo "  endpoint: ${ENDPOINT}"
  echo "  log:      ${LOGFILE}"
  echo "  new shells (or re-activation) pick up the endpoint automatically;"
  echo "  in the current shell: source ${SESSION_ENV}"
}

_do_stop() {
  if _instance_alive; then
    local pid
    pid=$(cat "$PIDFILE")
    local pgid
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    kill -TERM -- -"$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    sleep 3
    kill -KILL -- -"$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    echo "gem5 stopped (was pid ${pid})"
  else
    echo "gem5 not running"
  fi
  rm -f "$SESSION_ENV" "$PIDFILE"
}

_do_status() {
  if _instance_alive; then
    local dispatches=0
    [[ -f ${SESSION_ROOT}/dispatch-trace.jsonl ]] && \
      dispatches=$(wc -l < "${SESSION_ROOT}/dispatch-trace.jsonl")
    echo "gem5 RUNNING"
    echo "  pid:        $(cat "$PIDFILE")"
    echo "  endpoint:   ${ENDPOINT}"
    echo "  session:    ${SESSION_ROOT}"
    echo "  dispatches: ${dispatches}"
  else
    echo "gem5 STOPPED"
    [[ -f $LOGFILE ]] && echo "  last log: ${LOGFILE}"
  fi
}

case $cmd in
  start)   _do_start ;;
  stop)    _do_stop ;;
  restart) _do_stop; sleep 1; _do_start ;;
  status)  _do_status ;;
  *) echo "usage: gem5-session start|stop|restart|status [--accurate]" >&2; exit 2 ;;
esac
