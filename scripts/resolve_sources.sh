#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--online" ]]; then
  echo "usage: scripts/resolve_sources.sh --online" >&2
  exit 2
fi
cat >&2 <<'MSG'
P0-SRC-01 is a guarded operation. Resolve each official URL with
`git ls-remote --symref URL HEAD`, verify source trees/licenses, resolve the
official Hugging Face revision through HfApi, write SOURCE_LOCK.json with
immutable commit/tree hashes, then clone pristine lanes and create baseline
tags. This bootstrap entrypoint does not mutate sources.
MSG
echo "source resolution is the next checkpoint action; no sources were changed"
exit 3
