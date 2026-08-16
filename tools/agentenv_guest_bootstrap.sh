#!/usr/bin/env bash
# Install and validate one feature-local runtime bundle inside an AgentENV VM.
# The archive contains relative tar names rooted at /home/zhaosiying/amdgpu-sim.
set -euo pipefail

usage() {
  printf 'usage: %s ARCHIVE MANIFEST [NAMESPACE]\n' "$0" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage
ARCHIVE=$1
MANIFEST=$2
NAMESPACE=${3:-${AGENTENV_NAMESPACE:-}}

[[ -f "$ARCHIVE" ]] || { echo "missing archive: $ARCHIVE" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }
command -v python3 >/dev/null || { echo 'python3 is required' >&2; exit 1; }
command -v zstd >/dev/null || { echo 'zstd is required' >&2; exit 1; }
command -v tar >/dev/null || { echo 'tar is required' >&2; exit 1; }

if [[ $(id -u) -ne 0 ]]; then
  echo 'bundle extraction requires root so absolute guest paths retain ownership' >&2
  exit 1
fi

EXPECTED=$(python3 - "$MANIFEST" <<'PY'
import json, pathlib, sys
doc = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
value = doc.get("archive_sha256")
if not isinstance(value, str) or len(value) != 64:
    raise SystemExit("manifest has no archive_sha256")
print(value)
PY
)
ACTUAL=$(sha256sum "$ARCHIVE" | awk '{print $1}')
[[ "$ACTUAL" == "$EXPECTED" ]] || {
  echo "archive sha256 mismatch: $ACTUAL != $EXPECTED" >&2
  exit 1
}

umask 077
if [[ -n "$NAMESPACE" ]]; then
  case "$NAMESPACE" in
    (*[!A-Za-z0-9._-]*|'') echo "invalid namespace: $NAMESPACE" >&2; exit 1 ;;
  esac
  export TMPDIR=${TMPDIR:-/tmp/$NAMESPACE}
  export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/$NAMESPACE}
  export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/var/cache/$NAMESPACE}
  export HF_HOME=${HF_HOME:-$XDG_CACHE_HOME/huggingface}
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}
  install -d -m 700 "$TMPDIR" "$XDG_RUNTIME_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR"
fi

# Tar entries are relative and intentionally extract below /; the archive
# digest protects the transport without a second full read of the tree.
zstd -q -dc "$ARCHIVE" | tar -xpf - -C / --numeric-owner

GUEST_ROOT=$(python3 - "$MANIFEST" <<'PY'
import json, pathlib, sys
doc = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
root = doc.get("guest_root")
if not isinstance(root, str) or not root.startswith("/") or root == "/":
    raise SystemExit("manifest has invalid guest_root")
print(root)
PY
)
test -d "$GUEST_ROOT" || { echo "guest root missing after extraction: $GUEST_ROOT" >&2; exit 1; }

printf 'agentenv_bundle_ready=1\n'
printf 'agentenv_namespace=%s\n' "$NAMESPACE"
printf 'agentenv_guest_root=%s\n' "$GUEST_ROOT"
printf 'agentenv_archive_sha256=%s\n' "$ACTUAL"
