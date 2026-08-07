#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

mode=""
online=0
for arg in "$@"; do
  case "$arg" in
    --verify) mode=verify ;;
    --online) online=1 ;;
    *) echo "usage: scripts/resume.sh --verify [--online]" >&2; exit 2 ;;
  esac
done
[[ "$mode" == verify ]] || { echo "usage: scripts/resume.sh --verify [--online]" >&2; exit 2; }

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root_dir"
python3 scripts/verify_workspace.py --root "$root_dir"

if (( online )); then
  exec python3 scripts/source_manager.py verify-online
fi
