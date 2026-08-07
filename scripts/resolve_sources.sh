#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
if [[ "${1:-}" != "--online" ]]; then
  echo "usage: scripts/resolve_sources.sh --online" >&2
  exit 2
fi
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root_dir"
exec python3 scripts/source_manager.py resolve --output artifacts/source-resolution.json
